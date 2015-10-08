package nate.probschemas;

import java.io.File;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;

import edu.stanford.nlp.stats.ClassicCounter;
import edu.stanford.nlp.stats.Counter;
import nate.util.Pair;
import nate.muc.MUCKeyReader;
import nate.muc.KeyReader;
import nate.ProcessedData;
import nate.util.Directory;
import nate.util.HandleParameters;
import nate.util.Triple;
import nate.util.Util;
import nate.util.TreeOperator;

/**
 * Trains my graphical model for template learning.
 * Input is a directory that contains parsed/deps/events/ner files.
 * If run as a learner, it saves the model to disk at sampler-*.model.
 * 
 * Learner -topics <int> [-n <int>] [-d <int>] [-ir <int>] [-avg] -train <data-dir>,<data-dir>,etc.
 * Learner -model <filepath> -key <path> [-p <double>] [-m <int>] <data-dir>
 * 
 * LEARNING
 * -train   : The directory containing text processed files.
 * -plates  : The number of templates to learn.
 * -jplates : The number of junk templates to learn.
 * -topics  : The number of slots across all templates. (e.g., if 5 templates, 20 topics is 4 per template)
 * -jtopics : The number of junk slots to include in sampling.
 * -dtheta  : Use thetas per document, not as a single global distribution.
 * -n       : The number of Gibbs sampling iterations.
 * -d       : The number of training documents to use (if not given, all training docs used).
 * -ir      : The number of documents to retrieve (if not present, no IR is used).
 * -avg     : If present, sample the training set multiple times, infer, and average the results.
 * -isamp   : If present, use a saved sampler model, and infer using its sampled labels.
 * -noent   : If present, don't use NER features in the sampler.
 * -c       : Cutoff for dep counts to keep mention.
 * -cdoc    : Cutoff for verb counts to keep its mentions.
 * -sw      : Dirichlet smoothing parameter for words in the sampler.
 * -sd      : Dirichlet smoothing parameter for deps in the sampler.
 * -sv      : Dirichlet smoothing parameter for verbs in the sampler. Also, if this is used, it turns on the verbs variable in the graphical model.
 * -sf      : Dirichlet smoothing parameter for entity features in the sampler.
 * 
 * INFERENCE
 * -model  : The pretrained model to load. If given, runs inference instead of training.
 * -key    : The gold answers from MUC to evaluate against.
 * -p      : Inference...the minimum probability an entity must be labeled.
 * -m      : Inference...the max number of entities mapped to any given role.
 * 
 * EXAMPLES
 * java nate.probschemas.Learner -topics 25 -m 10 -ir 100 kidnap/
 *   - train a 25-topic model on kidnapping, first 10 docs, and with 100 docs from IR.
 * 
 */
public class Learner {
//  IRDocuments ir;
  List<List<TextEntity>> _trainDocsEntities;
  List<String> _trainDocsNames;
  List<List<TextEntity>> _testDocsEntities;
  List<String> _testDocsNames;
  ProcessedData _loadedTrainData;

  // Set to true to use GibbsSamplerWorkshop instead of the full model GibbsSamplerEntities
  boolean _workshop = false;
  
  boolean _sampleEntityModel = true;
  boolean _doIR              = false;
  boolean _learnAndInfer     = false;
  boolean _includeEntFeats	 = true;
  boolean _constrainInverseDeps   = true;
  boolean _thetasInDoc       = true;
  int _numIRDocs = 0;
  int _numTrainingDocs = Integer.MAX_VALUE;
  int _minDepCounts = 2;
  int _minDocCounts = 10;
  int _numTopics = 10;
  int _numTemplates = 0;  // if positive, then a nested template->topic->entity model is used
  int _numJunkTopics = 0;
  int _numJunkTemplates = 0;
  int _sampleSteps = 1000;
  double _inferMinProb = 0.95;
  int _inferMaxEntities = 3;
  int _inferMaxRolesPerSlot = Integer.MAX_VALUE; // After inference, our learned roles are mapped to gold slots. How many roles can go to the same slot?
  String _modelPath = null;
  boolean _inferBySampler = false;
  boolean _ignoreIsolated = false;
  String _trainDataDir = null;
  String _testDataDir = null;
  String _testKeyPath = null;
  String _modelOutDir = ".";
  String _modelOutName = null;
  boolean _skipCache = false; // if true, data is not read from nor written to the cache
  
  double _depSmoothing = -1.0, _wordSmoothing = -1.0, _featSmoothing = -1.0, _verbSmoothing = -1.0;
  boolean _debugOn = false;
  boolean _evaluateOnlyTemplateDocs = false; // if true, artificially hide documents in test set that aren't labeled
  boolean _evaluateIgnoreSchemas = false;    // if true, map any learned role to any slot and ignore schema clusterings
  
    //  final String[] _types = { "KIDNAP", "BOMBING", "ATTACK", "FORCED WORK STOPPAGE", "ROBBERY", "ARSON" };

  
  public Learner(String args[]) {
    HandleParameters params = new HandleParameters(args);
   
    if( args.length < 1 ) {
      System.out.println("Learner [-model <filepath>] [-topics <int>] [-n <int>] <data-dir>");
      System.exit(-1);
    }
    
    if( params.hasFlag("-topics") ) _numTopics        = Integer.parseInt(params.get("-topics"));
    if( params.hasFlag("-plates") ) _numTemplates     = Integer.parseInt(params.get("-plates"));
    if( params.hasFlag("-templates") ) _numTemplates  = Integer.parseInt(params.get("-templates"));
    if( params.hasFlag("-jtopics") ) _numJunkTopics   = Integer.parseInt(params.get("-jtopics"));
    if( params.hasFlag("-jplates") ) _numJunkTemplates= Integer.parseInt(params.get("-jplates"));
    if( params.hasFlag("-dtheta") || params.hasFlag("-dthetas") ) _thetasInDoc      = true;
    if( params.hasFlag("-n") )      _sampleSteps      = Integer.parseInt(params.get("-n"));
    if( params.hasFlag("-d") )      _numTrainingDocs  = Integer.parseInt(params.get("-d"));
    if( params.hasFlag("-c") )      _minDepCounts     = Integer.parseInt(params.get("-c"));
    if( params.hasFlag("-cdoc") )   _minDocCounts     = Integer.parseInt(params.get("-cdoc"));
    if( params.hasFlag("-sd") )     _depSmoothing 	  = Double.parseDouble(params.get("-sd"));
    if( params.hasFlag("-sw") )     _wordSmoothing 	  = Double.parseDouble(params.get("-sw"));
    if( params.hasFlag("-sf") )     _featSmoothing    = Double.parseDouble(params.get("-sf"));
    if( params.hasFlag("-sv") )     _verbSmoothing    = Double.parseDouble(params.get("-sv"));
    if( params.hasFlag("-noent"))   _includeEntFeats  = false;
    if( params.hasFlag("-noconstraints")) _constrainInverseDeps = false;
    if( params.hasFlag("-avg") )    _learnAndInfer    = true;
    if( params.hasFlag("-isamp") )  _inferBySampler   = true;
    if( params.hasFlag("-isolated") ) _ignoreIsolated = true;
    if( params.hasFlag("-model") )  _modelPath        = params.get("-model");
    if( params.hasFlag("-p") )      _inferMinProb     = Double.parseDouble(params.get("-p"));
    if( params.hasFlag("-m") )      _inferMaxEntities = Integer.parseInt(params.get("-m"));
    if( params.hasFlag("-roles") )	_inferMaxRolesPerSlot = Integer.parseInt(params.get("-roles"));
    if( params.hasFlag("-ir") && Integer.parseInt(params.get("-ir")) > 0 ) {  _doIR = true; _numIRDocs = Integer.parseInt(params.get("-ir"));  }
    if( params.hasFlag("-debug") )  _debugOn          = true;
    if( params.hasFlag("-evaleasy") )          _evaluateOnlyTemplateDocs = true;
    if( params.hasFlag("-evalignoreschemas") ) _evaluateIgnoreSchemas = true;
    if( params.hasFlag("-out") )    _modelOutDir      = params.get("-out");
    if( params.hasFlag("-outmodel") ) _modelOutName   = params.get("-outmodel");
    if( params.hasFlag("-test") )   _testDataDir      = params.get("-test");
    if( params.hasFlag("-testkey")) _testKeyPath      = params.get("-testkey");
    if( params.hasFlag("-train") )   _trainDataDir      = params.get("-train");
    if( params.hasFlag("-skipcache")) _skipCache      = true;
    if( params.hasFlag("-workshop"))  _workshop       = true;

    System.out.println("smoothing: " + _wordSmoothing + "\t" + _depSmoothing + "\t" + _featSmoothing);
    System.out.println("Use Entity Model:\t" + _sampleEntityModel);
    System.out.println("Thetas per document?\t" + _thetasInDoc);
    System.out.println("Num topics:\t\t" + _numTopics);
    System.out.println("Num junk topics:\t" + _numJunkTopics);
    System.out.println("Num templates:\t\t" + _numTemplates);
    System.out.println("Num junk templates:\t" + _numJunkTemplates);
    System.out.println("Num sampling steps:\t" + _sampleSteps);
    System.out.println("Num training docs:\t" + _numTrainingDocs);
    System.out.println("Min dep counts:\t" + _minDepCounts);
    System.out.println("IR turned " + (_doIR ? "on" : "off"));
    System.out.println("Num IR docs:\t\t" + _numIRDocs);
    if( _modelPath != null )
      System.out.println("Model:\t\t" + _modelPath);
    if( _modelPath != null || _learnAndInfer ) {
      System.out.println("Inference min P():\t\t" + _inferMinProb);
      System.out.println("Inference max num:\t\t" + _inferMaxEntities);
      System.out.println("Mapping num roles per slot:\t" + _inferMaxRolesPerSlot);
      System.out.println("Gold Templates:\t\t" + _testKeyPath);
    }
    
    // Load the IR Lucene index.
//    if( _doIR  ) ir = new IRDocuments();
    
    // Load the given training data.
    if( _trainDataDir != null ) 
    	load(_trainDataDir, false);
    System.out.println("Train dir:\t" + _trainDataDir);
    
    // Load the test data, if given.
    if( _testDataDir != null ) {
    	load(_testDataDir, true);
    	if( _testKeyPath == null ) {
    		System.out.println("WARNING: no test key given (-testkey), but test directory was given (-test).");
//    		System.exit(1);
    	}
    }
    System.out.println("Test dir:\t" + _testDataDir);
    System.out.println("Test key:\t" + _testKeyPath);
    System.out.println("Eval easy:\t" + _evaluateOnlyTemplateDocs);
    System.out.println("Eval ignore schemas:\t" + _evaluateIgnoreSchemas);
    
    if( _workshop )
      System.out.println("*************************************\nRunning Workshop Code, not full model!\n****************************\n");
  }

  /**
   * Populates the _trainDocsEntities variable based on this data directory.
   * @param dataDir Path to a directory that contains the four preprocessed text files.
   */
  public void load(String dataDir, boolean loadIntoTest) {
	  // Initialize the lists.
	  if( !loadIntoTest ) {
		  _trainDocsEntities = new ArrayList<List<TextEntity>>();
		  _trainDocsNames = new ArrayList<String>();
	  } else {
		  _testDocsEntities = new ArrayList<List<TextEntity>>();
		  _testDocsNames = new ArrayList<String>();
	  }

	  // User can specify more than one directory to load, separated by commas.
	  String[] dirs = dataDir.split(",");
	  for( String dir : dirs ) {
		  System.out.println("Loading from directory " + dir);
		  String parsesFile = dir + File.separator + Directory.nearestFile("parse", dir, "idf");
		  String depsFile   = dir + File.separator + Directory.nearestFile("deps", dir, "idf");
		  String eventsFile = dir + File.separator + Directory.nearestFile("events", dir, "idf");
		  String nerFile    = dir + File.separator + Directory.nearestFile("ner", dir, "idf");

		  // If we are given a directory, not a file.
		  if( (new File(parsesFile)).isDirectory() ) {
			  System.out.println("ERROR: file in load() is a directory, expected a file: " + parsesFile);
			  System.exit(1);
		  } 

		  // Load them!
		  if( loadIntoTest ) {
		  	loadFile(parsesFile, depsFile, eventsFile, nerFile, _testDocsNames, _testDocsEntities, loadIntoTest);
			  System.out.println("Num test docs = " + _testDocsNames.size() + " and " + _testDocsEntities.size());
		  }
		  else {
		  	loadFile(parsesFile, depsFile, eventsFile, nerFile, _trainDocsNames, _trainDocsEntities, loadIntoTest);
			  System.out.println("Num training docs = " + _trainDocsNames.size() + " and " + _trainDocsEntities.size());
		  }
	  }
  }

  /**
   * Load files of Entities. This fills the global _trainDocsEntities or _testDocsEntities variable
   * depending on the boolean parameter loadIntoTest.
   * @param parsesPath Directory or single file or parses.
   * @param depsPath
   * @param eventsPath
   * @param nerPath
   * @param docnames A list of document names. This function appends to it.
   * @param allentities A list of document entity lists. This function appends to it.
   * @param loadIntoTest If true, then load data into global _testDocsEntities. If false, _trainDocsEntities.
   */
  public void loadFile(String parsesPath, String depsPath, String eventsPath, String nerPath, List<String> docnames, List<List<TextEntity>> allentities, boolean loadIntoTest) {
	  System.out.println("parses: " + parsesPath + " and deps: " + depsPath + " and events: " + eventsPath + "and ner: " + nerPath);
	  DataSimplifier simplify = new DataSimplifier(_minDepCounts, _minDocCounts);

	  // Read the data files from disk (IR needs this later).
	  if( !loadIntoTest )
		  _loadedTrainData = new ProcessedData(parsesPath, depsPath, eventsPath, nerPath);

	  // Check the cache of this file. Read from there if we already processed it!
	  String cachePrefix = (loadIntoTest ? "test-" : "train-") + parsesPath;
	  Pair<List<String>,List<List<TextEntity>>> cached = simplify.getCachedEntityList(cachePrefix);
	  if( !_skipCache && cached != null ) {
		  allentities.addAll(cached.second());
		  docnames.addAll(cached.first());
		  System.out.println("Cached number of docs in entity list: " + cached.second().size());
		  System.out.println("Cached doc names: " + cached.first());
		  System.out.println("Total doc names now " + docnames.size() + " with " + allentities.size() + " docs for entities (these numbers should be equal!).");
	  }
	  // Not cached yet.
	  else {
		  // Get the tokens and dependencies from this file.
		  if( loadIntoTest ) {
			  ProcessedData thedata = new ProcessedData(parsesPath, depsPath, eventsPath, nerPath);
			  List<String> myDocnames = new ArrayList<String>();
			  List<List<TextEntity>> docEntities = simplify.getEntityList(thedata, myDocnames, Integer.MAX_VALUE);
			  docnames.addAll(myDocnames);
			  allentities.addAll(docEntities);
			  System.out.println("Loaded data into test.");
			  // Write to cache.
			  if( !_skipCache ) simplify.writeToCache(cachePrefix, myDocnames, docEntities);
			  System.out.println("Past the cache writing.");
		  }
		  else {
			  _loadedTrainData.nextStory();
			  List<String> myDocnames = new ArrayList<String>();
			  List<List<TextEntity>> docEntities = simplify.getEntityList(_loadedTrainData, myDocnames, Integer.MAX_VALUE); 
			  docnames.addAll(myDocnames);
			  allentities.addAll(docEntities);
			  System.out.println("Loaded data into training.");
			  // Write to cache.
			  if( !_skipCache ) simplify.writeToCache(cachePrefix, myDocnames, docEntities);
			  System.out.println("Past the cache writing.");
		  }
	  }
  }
  
  /**
   * 
   * @param ndocs The number of documents to retrieve and add to training.
   */
  private void expandDocumentsWithIR(List<String> docsNames, List<List<TextEntity>> docsEntities, int ndocs) {
    if( ndocs <= 0 ) return;
    
    System.out.println("expandDocumentsWithIR() ndocs=" + ndocs);
    KeywordDetector detector = new KeywordDetector();
    _loadedTrainData.reset();
    
    Counter<String> allcounts = new ClassicCounter<String>();

    // Sum up the keywords seen in each document.
    for( String doc : docsNames ) {
      _loadedTrainData.nextStory(doc);
      List<String> keywords = detector.getKeywords(TreeOperator.stringsToTrees(_loadedTrainData.getParseStrings()), _loadedTrainData.getDependencies());
      for( String key : keywords )
        allcounts.incrementCount(key);
    }
    
    // Weight keywords by IDF scores.
    for( String key : allcounts.keySet() ) {
      allcounts.setCount(key, allcounts.getCount(key) * detector.idf.get("v-" + key));
    }
    List<String> sortedKeys = Util.sortCounterKeys(allcounts);
    
    // Sort and save the top 10 keys.
    List<String> topKeys = new ArrayList<String>();
    int nn = 0;
    for( String key : sortedKeys ) { 
      System.out.println("key: " + key + "\t" + allcounts.getCount(key));
      topKeys.add(key);
      if( nn++ >= 10 ) break;
    }
    
    // Build the query list, duplicate keys with high scores for added weight.
    double min = allcounts.getCount(topKeys.get(topKeys.size()-1));
    List<String> query = new ArrayList<String>();
    for( String key : topKeys ) {
      int times = (int)(allcounts.getCount(key) / min);
      for( int xx = 0; xx < times; xx++ )
        query.add(key);
    }
    System.out.println("query: " + query);

    // Make the IR query, append the results to the global variables (docsNames destructively, docsEntities later).
//    List<List<TextEntity>> irDocsEntities = ir.retrieveDocumentsAsEntities(query, ndocs, docsNames);
//    docsEntities.addAll(irDocsEntities);
    
    System.out.println("Finished IR, new docs added to locally given docsEntities.");
  }

  public GibbsSamplerEntities createSampler(List<String> docnames, List<List<TextEntity>> docsEntities) {
    GibbsSamplerEntities sampler;
    if( _workshop ) {
      sampler = new GibbsSamplerWorkshop(_numTopics, _numJunkTopics, _numTemplates, _numJunkTemplates);
    }
    else {
      sampler = new GibbsSamplerEntities(_numTopics, _numJunkTopics, _numTemplates, _numJunkTemplates);
    }

    if( _wordSmoothing >= 0.0 ) sampler.wSmoothing = _wordSmoothing;
    if( _depSmoothing >= 0.0 ) sampler.depSmoothing = _depSmoothing;
    if( _featSmoothing >= 0.0 ) sampler.featSmoothing = _featSmoothing;
    if( _verbSmoothing > 0.0 ) { sampler.includeVerbs = true; sampler.verbSmoothing = _verbSmoothing; }
    sampler.includeEntityFeatures = _includeEntFeats;
    sampler.constrainInverseDeps = _constrainInverseDeps;
    sampler.thetasInDoc = _thetasInDoc;
    sampler.initializeModelFromData(docnames, docsEntities);
    //      sampler.runSampler(_sampleSteps/2);
    //      sampler.printWordDistributionsPerTopic();

    return sampler;
  }
  
  /**
   * Use the global entity list per doc, and pull out a random number of documents (num docs)
   * that will be used for learning. Put those num documents into the given fillEntities list.
   * @param num The number of random documents to select.
   * @param fillNames The list of document names selected.
   * @param fillEntities The entity lists for each document selected.
   */
  private void getRandomDocsFromTrain(int num, List<String> fillNames, List<List<TextEntity>> fillEntities) {
    // Asked for more docs than we have, so just return them all.
    if( num >= _trainDocsNames.size() ) {
      for( int ii = 0; ii < _trainDocsNames.size(); ii++ ) {
        fillNames.add(_trainDocsNames.get(ii));
        fillEntities.add(_trainDocsEntities.get(ii));
      }
      return;
    }
    
    // Asked for less docs than we have, so randomly sample.
    Random rand = new Random(); 
    Set<Integer> docids = new HashSet<Integer>();
    while( docids.size() < num ) {
      int draw = rand.nextInt(_trainDocsNames.size());
      while( docids.contains(draw) )
        draw = rand.nextInt(_trainDocsNames.size());
      docids.add(draw);
    }
    
    System.out.println("Wanted " + num + " random docs, drew " + docids.size() + " docs.");
    
    for( int ii = 0; ii < _trainDocsNames.size(); ii++ ) {
      if( docids.contains(ii) ) {
        fillNames.add(_trainDocsNames.get(ii));
        fillEntities.add(_trainDocsEntities.get(ii));
      }
    }
  }

  /**
   * Typically names/entities are training documents, and newNames/newEntities are test documents.
   * This function adds all test to the train. Sometimes the test and the train are the same thing, so
   * this function does not add any test docs that are already in train.
   */
  private void addDocsNoRepeats(List<String> names, List<List<TextEntity>> entities, List<String> newNames, List<List<TextEntity>> newEntities) {
    if( newNames != null ) {
      // Loop over each new document.
      for( int ii = 0; ii < newNames.size(); ii++ ) {
        String doc = newNames.get(ii);
        boolean exists = false;
        for( String nn : names ) 
          if( nn.equalsIgnoreCase(doc) )
            exists = true;
        // If the document doesn't already exist in the names, add it.
        if( !exists ) {
          names.add(doc);
          entities.add(newEntities.get(ii));
        }
      }
    }
  }
  
  private KeyReader getAnswerKey() {
    // Load the answer key for MUC or Corporate Acq.
    KeyReader answerKey = null;
    String keypath = _testKeyPath;
    if( keypath != null && keypath.contains("muc") ) 
      answerKey = new MUCKeyReader(keypath);
    else 
      System.out.println("ERROR (Learner.java) getAnswerKey() only runs 'muc'");
    return answerKey;
  }
  
  /**
	 * Sort the given array of topic probabilities, and return the topic IDs in order.
	 * The cutoff parameter only returns topic IDs who had a high enough probability.
	 */
	private List<Integer> getSortedTopics(double[] topicProbs, double cutoff) {
		Counter<Integer> counter = new ClassicCounter<Integer>();
		for( int ii = 0; ii < topicProbs.length; ii++ ) {
			if( topicProbs[ii] >= cutoff )
				counter.incrementCount(ii, topicProbs[ii]);
		}
		List<Integer> sortedTopics = Util.sortCounterKeys(counter);
		return sortedTopics;
	}

	/**
	 * This function takes a list of entities with topic labels, and counts all the topics.
	 * It then removes topics that are singletons in their template. Only topics who have sibling topics
	 * in the document are kept.
	 * @param doc A list of entities in a document.
	 * @param sampler A sampler with a templates->topics->entity nested model.
	 */
	private void removeIsolatedTemplateLabels(List<TextEntity> doc, GibbsSamplerEntities sampler) {
		// Count how many times each template is seen.
		Counter<Integer> templateCounts = new ClassicCounter<Integer>();
		for( TextEntity ent : doc ) {
			if( ent.hasALabel() )
				for( Integer topic : ent.getLabels() ) {
					int template = topic / sampler.numTemplates;
					templateCounts.incrementCount(template);
				}
		}

		// Remove any topic whose template was only seen once.
		for( TextEntity ent : doc ) {
			if( ent.hasALabel() ) {
				List<Integer> keeptopics = new ArrayList<Integer>();
				for( Integer topic : ent.getLabels() ) {
					int template = topic / sampler.numTemplates;
					if( templateCounts.getCount(template) > 1 )
						keeptopics.add(topic);
				}
				// Put the good ones back in.
				ent.clearLabels();
				for( Integer keep : keeptopics )
					ent.addLabel(keep);
			}
		}
	}
	
	/**
	 * Print the list of entities with their template/role label.
	 * @param doc List of TextEntity objects, representing all entities in a single document.
	 */
	private void printEntityLabels(List<TextEntity> doc) {
		for( TextEntity ent : doc ) {
		  if( ent.hasALabel() ) System.out.print(ent.getLabels() + "\t");
			else System.out.print("NONE\t");
			System.out.println(ent);
		}
	}

	/**
	 * The sampler already labeled everything, so use its labels to evaluate performance.
	 * Don't run a separate inference step.
	 * 
	 * @return Two double arrays. Each has three numbers: precision, recall, F1
	 *          The first array is the main schema->template mapping results.
	 *          The second array is the greedy role->slot (ignore schemas) mapping.
	 */
	public Triple inferUsingSampledData(GibbsSamplerEntities sampler) {
	  System.out.println("Infer from Sampled Data!");

	  // Load the sampled data.
	  if( sampler == null ) {
//	    sampler = new GibbsSamplerEntities();
	    sampler = GibbsSamplerEntities.fromFile(_modelPath);
	    System.out.println("*** Loaded Sampler from Disk ***");
	    sampler.printWordDistributionsPerTopic();
	    System.out.println();
	  }

	  // Evaluate
	  KeyReader answerKey = getAnswerKey();
	  EvaluateModel evaluator = new EvaluateModel(sampler.numTopics, answerKey);
	  evaluator._debugOn = _debugOn;
	  evaluator._evaluateOnTemplateDocsOnly = _evaluateOnlyTemplateDocs;

	  // Set the appropriate test set (train set often is the test set during development)
	  List<List<TextEntity>> testEntities = _testDocsEntities;
	  List<String> testDocNames = _testDocsNames;
	  System.out.println("Inferring from " + testEntities.size() + " docs.");

	  //    final double[] probs = { .98, .9, .8, .7, .6, .5, .4, .3 };
	  //    final double[] probs = { .98, .9, .7, .5, .3, .2, .1, .05, .02 };
	  final double[] probs = { .1 };
	  double[][] prf1s = new double[probs.length][];
	  double[][] prf1sGreedy = new double[probs.length][];
	  double[][] prf1sGoldGreedy = new double[probs.length][];
	  int jj = 0;
	  for( double prob : probs ) {

	    // Grab the labeled entities from the sampler!
	    // Look up the z topics in the sampler to add the labels.
	    int[][] zs = sampler.getZs();
	    int testdoci = 0;
	    for( List<TextEntity> doc : testEntities ) {
	      int samplerDoci = sampler.docnameToIndex(testDocNames.get(testdoci));
	      if( samplerDoci == -1 ) {
	        System.out.println("Test document " + testDocNames.get(testdoci) + " does not exist in the sampled model.");
	        System.exit(-1);
	      }
	      for( int entityi = 0; entityi < doc.size(); entityi++ ) {
	        int z = zs[samplerDoci][entityi];
	        sampler.unlabel(samplerDoci, entityi);
	        double[] topicProbs = sampler.getTopicDistribution(samplerDoci, entityi);

	        List<Integer> sortedTopics = getSortedTopics(topicProbs, prob);

	        int bestTopic = 0;
	        if( sortedTopics.size() > 0 ) {
	          bestTopic = sortedTopics.get(0);

	          // If we enforce matches must be with top verbs/words in a template.
	          //    				TextEntity targetEntity = doc.get(entityi);
	          //    				List<String> topTemplateVerbs = sampler.getTopVerbsInTemplate(bestTopic, 2, 0.1);
	          //    				boolean top = false;
	          //    				System.out.println("targetEntity: " + targetEntity);
	          //    				for( int ii = 0; ii < targetEntity.numMentions(); ii++ ) {
	          //    					String dep = targetEntity.getMentionDependency(ii);
	          //    					String governor = dep.substring(dep.indexOf("--")+2);
	          //    					System.out.println("Comparing " + governor);
	          //    					if( Util.findInListOfStrings(topTemplateVerbs, governor) )
	          //    						top = true;
	          //    				}
	          //    				if( top )
	          //    					doc.get(entityi).addLabel(bestTopic);
	          //    				else {
	          //    					System.out.println("LEARNER: not adding non-top entity label: " + targetEntity);
	          //    					System.out.println("LEARNER: top verbs topic " + bestTopic + ": " + topTemplateVerbs);
	          //    				}

	          // Label it for evaluation
	          doc.get(entityi).addLabel(bestTopic);
	        }

	        sampler.relabel(samplerDoci, entityi, z);
	      }

	      if( _ignoreIsolated )
	        removeIsolatedTemplateLabels(doc, sampler);
	      
	      System.out.println("---- DOCUMENT LABELS (" + testDocNames.get(testdoci) + ") ----");
	      printEntityLabels(doc);
	      testdoci++;
	    }

	    // If we have an answer key, calculate the P/R/F1 scores.
	    if( evaluator != null && evaluator._answerKey != null ) {
	      // Save our guesses to the evaluator.
	      evaluator.setGuesses(testDocNames, testEntities);

	      // Find the best topic/slot mapping.
	      System.out.println("** Evaluate Entities **");
	      double[] PRF1 = null;
	      // If no templates, then it is a flat slot-only mapping.
	      if( sampler.numTemplates == 0 ) 
	        //    		PRF1 = evaluator.evaluateSlotsGreedy(_inferMaxRolesPerSlot);
	        PRF1 = evaluator.evaluateSlotsBestSingleRoleEachSlot();
	      else
	        //    		PRF1 = evaluator.evaluateSlotsAsTemplatesGreedy(_inferMaxRolesPerSlot, sampler.numTemplates);
	        //    		PRF1 = evaluator.evaluateSlotsBestSingleRoleEachSlot();
	        PRF1 = evaluator.evaluateSlotsBestSchemaForEachMUCType(sampler.numTemplates);
	      prf1s[jj] = PRF1;

	      System.out.printf("Results:\tp=%.3f\tr=%.3f\tf1=%.2f\n", PRF1[0], PRF1[1], PRF1[2]);

	      if( _evaluateIgnoreSchemas ) {
	        System.out.println("Also running no-schema mapping...any role to any slot.");
	        double[] ignoredPRF1 = evaluator.evaluateSlotsBestSingleRoleEachSlot();
	        System.out.printf("IGNORE SCHEMA Results:\tp=%.3f\tr=%.3f\tf1=%.2f\n", ignoredPRF1[0], ignoredPRF1[1], ignoredPRF1[2]);

	        System.out.println("Also running no-schema mapping greedily...any role to any slot, but only one role per slot max.");
	        double[] ignoredGreedyPRF1 = evaluator.evaluateSlotsGreedy(4);
	        System.out.printf("IGNORE SCHEMA GREEDY Results:\tp=%.3f\tr=%.3f\tf1=%.2f\n", ignoredGreedyPRF1[0], ignoredGreedyPRF1[1], ignoredGreedyPRF1[2]);
	        prf1sGreedy[jj] = ignoredGreedyPRF1;

	        System.out.println("Also running GOLD no-schema mapping greedily...any role to any slot, but only one role per slot max.");
	        boolean prev = evaluator._evaluateOnTemplateDocsOnly;
	        evaluator._evaluateOnTemplateDocsOnly = true;
	        double[] ignoredGoldGreedyPRF1 = evaluator.evaluateSlotsGreedy(4);
	        System.out.printf("IGNORE SCHEMA GREEDY GOLD Results:\tp=%.3f\tr=%.3f\tf1=%.2f\n", ignoredGoldGreedyPRF1[0], ignoredGoldGreedyPRF1[1], ignoredGoldGreedyPRF1[2]);
	        prf1sGoldGreedy[jj] = ignoredGoldGreedyPRF1;
	        evaluator._evaluateOnTemplateDocsOnly = prev;
	      }
	    }
	    
	    jj++;
	  }

	  // If we had an answer key, print out the P/R/F1 scores.
    if( evaluator != null && evaluator._answerKey != null ) {
      System.out.println("** Evaluated All Entities **");
      jj = 0;
      for( double[] prf1 : prf1s ) {
        System.out.printf("p=%.2f\tall\tprec=%.3f\trecall=%.3f\tf1=%.2f\n", probs[jj], prf1[0], prf1[1], prf1[2]);
        jj++;
      }
      return new Triple(prf1s[prf1s.length-1], prf1sGreedy[prf1sGreedy.length-1], prf1sGoldGreedy[prf1sGreedy.length-1]);
    }

    // Else we just labeled the documents, and no evaluation.
    else return null;
	}
  
  /**
   * This method trains a model with gibbs sampling on a random subset of the data, and tests on the other.
   * It then repeats and averages the performance scores. The learned models are never saved to disk.
   */
  public void learnAndInferAvg() {
    System.out.println("Learn and Infer!");
    int numRuns = 30, runi = 0;;
    if( _numTrainingDocs >= _trainDocsNames.size() ) {
      numRuns = 5;
      if( _trainDocsNames.size() < 30 ) numRuns = 10; // if it is a very small set, let's just run it more for good measure
      
      System.out.println("Running only " + numRuns + " times because only " + _trainDocsNames.size() + " training docs, but you wanted " + _numTrainingDocs + ".");
      _numTrainingDocs = _trainDocsNames.size();
    }
    double[][] allruns = new double[numRuns][];

    // Load the answer key for MUC or Corporate Acq.
    KeyReader answerKey = getAnswerKey();
    
    // Set the appropriate test set (train set often is the test set during development)
    List<List<TextEntity>> testEntities = _trainDocsEntities;
    List<String> testDocnames = _trainDocsNames;
    if( _testDocsEntities != null ) testEntities = _testDocsEntities;
    if( _testDocsEntities != null ) testDocnames = _testDocsNames;
                                    
    // Sum up all the F1 scores and then average them!
    double[] sumPRF1 = { 0, 0, 0 };
    for( int run = 0; run < numRuns; run++ ) {
      System.out.println("Learn and Infer run " + run);

      // Get a random subset of the data.
      List<String> idocsNames = new ArrayList<String>();
      List<List<TextEntity>> idocsEntities = new ArrayList<List<TextEntity>>();
      getRandomDocsFromTrain(_numTrainingDocs, idocsNames, idocsEntities);
      Inference.clearEntityLabels(idocsEntities);

      System.out.println("rand docs: " + idocsNames);

      // Expand with IR.
      expandDocumentsWithIR(idocsNames, idocsEntities, _numIRDocs);
      System.out.println("IR expanded: " + idocsNames.size() + " doc names and " + idocsEntities.size() + " docs with entities.");

      // Learn
      System.out.println("Time to learn!");
      //        GibbsSamplerEntities sampler = new GibbsSamplerEntities(_numTopics);
      //        sampler.initializeModelFromData(idocsEntities);
      GibbsSamplerEntities sampler = createSampler(idocsNames, idocsEntities);
      sampler.runSampler(_sampleSteps);
      sampler.printWordDistributionsPerTopic();

      // Infer
      Inference infer = new Inference((GibbsSamplerEntities)sampler, _inferMaxEntities, _inferMinProb);
      EvaluateModel evaluator = new EvaluateModel(infer.sampler.numTopics, answerKey);
      evaluator._debugOn = _debugOn;

      // Label the global entities (not IR docs) with slots based on the learned model probabilities.
      int name = 0;
      Inference.clearEntityLabels(testEntities);
      for( List<TextEntity> doc : testEntities ) {
        System.out.println("label entities doc " + testDocnames.get(name++));
        infer.labelEntities(doc, _debugOn);
      }
      // Save our guesses to the evaluator.
      evaluator.setGuesses(testDocnames, testEntities);
      System.out.println("Set guesses on " + testDocnames.size() + " docs and " + testEntities.size() + " entities.");

      // Find the best topic/slot mapping.
      System.out.println("** Evaluate Entities **");
      double[] avgPRF1 = evaluator.evaluateSlotsGreedy(_inferMaxRolesPerSlot);
      allruns[runi++] = avgPRF1;
      for( int ii = 0; ii < sumPRF1.length; ii++ )
        if( !Double.isNaN(avgPRF1[ii]) ) 
          sumPRF1[ii] += avgPRF1[ii];
    }

    // Average.
    for( int ii = 0; ii < sumPRF1.length; ii++ )
      sumPRF1[ii] /= numRuns;
    for( int ii = 0; ii < allruns.length; ii++ )
      System.out.printf("run %d:\tp=%.3f\tr=%.3f\tf1=%.2f\n", ii, allruns[ii][0], allruns[ii][1], allruns[ii][2]);
    System.out.printf("Average of " + numRuns + " runs:\tp=%.3f\tr=%.3f\tf1=%.2f\n", sumPRF1[0], sumPRF1[1], sumPRF1[2]);
  }
  
  
  public void learnAndInferSampledAvg() {
    System.out.println("Learn and Infer By Sampling!");
    int numRuns = 10, runi = 0;;
    double[][] allruns = new double[numRuns][];
    double[][] allrunsGreedy = new double[numRuns][];
    double[][] allrunsGoldGreedy = new double[numRuns][];
    
    // Sum up all the F1 scores and then average them!
    double[] sumPRF1       = { 0, 0, 0 };
    double[] sumPRF1Greedy = { 0, 0, 0 };
    double[] sumPRF1GoldGreedy = { 0, 0, 0 };
    for( int run = 0; run < numRuns; run++ ) {
      System.out.println("Learn and Infer run " + run);

      // Get a random subset of the training data.
      List<String> idocsNames = new ArrayList<String>();
      List<List<TextEntity>> idocsEntities = new ArrayList<List<TextEntity>>();
      getRandomDocsFromTrain(_numTrainingDocs, idocsNames, idocsEntities);
      // Now add all the test docs.
      addDocsNoRepeats(idocsNames, idocsEntities, _testDocsNames, _testDocsEntities);
//      idocsEntities.addAll(_testDocsEntities);
//      idocsNames.addAll(_testDocsNames);
      Inference.clearEntityLabels(idocsEntities);

      System.out.println("rand docs: " + idocsNames);

      // Learn
      System.out.println("Time to learn!");
      GibbsSamplerEntities sampler = createSampler(idocsNames, idocsEntities);    
      sampler.runSampler(_sampleSteps);
      sampler.printWordDistributionsPerTopic();
      
      // Infer
      Triple scores = inferUsingSampledData(sampler);
      // Normal evaluation
      double[] prf1 = (double[])scores.first();
      allruns[runi] = prf1;
      for( int ii = 0; ii < sumPRF1.length; ii++ )
        if( !Double.isNaN(prf1[ii]) ) 
          sumPRF1[ii] += prf1[ii];
      // Greedy evaluation
      prf1 = (double[])scores.second();
      allrunsGreedy[runi] = prf1;
      for( int ii = 0; ii < sumPRF1Greedy.length; ii++ )
        if( !Double.isNaN(prf1[ii]) ) 
          sumPRF1Greedy[ii] += prf1[ii];
      // Gold Doc Greedy evaluation
      prf1 = (double[])scores.third();
      allrunsGoldGreedy[runi] = prf1;
      for( int ii = 0; ii < sumPRF1GoldGreedy.length; ii++ )
        if( !Double.isNaN(prf1[ii]) ) 
          sumPRF1GoldGreedy[ii] += prf1[ii];
      
      runi++;
    }

    // Average.
    for( int ii = 0; ii < sumPRF1.length; ii++ )
      sumPRF1[ii] /= numRuns;
    for( int ii = 0; ii < allruns.length; ii++ )
      System.out.printf("run %d:\tp=%.3f\tr=%.3f\tf1=%.2f\n", ii, allruns[ii][0], allruns[ii][1], allruns[ii][2]);
    System.out.printf("Average of " + numRuns + " runs:\tp=%.3f\tr=%.3f\tf1=%.2f\n", sumPRF1[0], sumPRF1[1], sumPRF1[2]);

    System.out.println("Greedy Evaluation Results");
    for( int ii = 0; ii < sumPRF1Greedy.length; ii++ )
      sumPRF1Greedy[ii] /= numRuns;
    for( int ii = 0; ii < allrunsGreedy.length; ii++ )
      System.out.printf("run %d:\tp=%.3f\tr=%.3f\tf1=%.2f\n", ii, allrunsGreedy[ii][0], allrunsGreedy[ii][1], allrunsGreedy[ii][2]);
    System.out.printf("Average of " + numRuns + " runs:\tp=%.3f\tr=%.3f\tf1=%.2f\n", sumPRF1Greedy[0], sumPRF1Greedy[1], sumPRF1Greedy[2]);

    System.out.println("Gold-Docs Greedy Evaluation Results");
    for( int ii = 0; ii < sumPRF1GoldGreedy.length; ii++ )
      sumPRF1GoldGreedy[ii] /= numRuns;
    for( int ii = 0; ii < allrunsGoldGreedy.length; ii++ )
      System.out.printf("run %d:\tp=%.3f\tr=%.3f\tf1=%.2f\n", ii, allrunsGoldGreedy[ii][0], allrunsGoldGreedy[ii][1], allrunsGoldGreedy[ii][2]);
    System.out.printf("Average of " + numRuns + " runs:\tp=%.3f\tr=%.3f\tf1=%.2f\n", sumPRF1GoldGreedy[0], sumPRF1GoldGreedy[1], sumPRF1GoldGreedy[2]);
}

	/**
   * Main function that is called after load() to run the gibbs sampler.
   * @return The Sampler after it finished running.
   */
  public GibbsSamplerEntities learn() {
    System.out.println("Time to learn!");
        
    // Learning mode with IR.
    /*
    if( _doIR && _numIRDocs > 0  ) {
      System.out.println("Base MUC docs: " + _trainDocsNames);
      
      // Load the IR Lucene index.
      ir = new IRDocuments();
      expandDocumentsWithIR(_trainDocsNames, _trainDocsEntities, _numIRDocs);
      System.out.println("Training set now has " + _trainDocsEntities.size() + " docs.");
    }
    */
    
    // Sanity check.
    if( _trainDocsNames.size() != _trainDocsEntities.size() ) {
      System.out.println("ERROR: names and entities lists not the same size: " + _trainDocsNames.size() + " " + _trainDocsEntities.size());
      System.exit(-1);
    }
    
    // Select random subset of the training set (if desired).
    List<String> docsNames = new ArrayList<String>();
    List<List<TextEntity>> docsEntities = new ArrayList<List<TextEntity>>();
    getRandomDocsFromTrain(_numTrainingDocs, docsNames, docsEntities);

    // Create the sampler.
    GibbsSamplerEntities sampler = createSampler(docsNames, docsEntities);    
    sampler.runSampler(_sampleSteps);
    sampler.printWordDistributionsPerTopic();
    if( _modelOutName != null )
      sampler.toFile(_modelOutDir + File.separator + _modelOutName);
    else {
      System.out.println(_trainDataDir + " stripped to " + Directory.lastSubdirectory(_trainDataDir));
      sampler.toFile(_modelOutDir + File.separator + "sampler-" + Directory.lastSubdirectory(_trainDataDir) + 
      		(_numTrainingDocs == Integer.MAX_VALUE ? "" : _numTrainingDocs) + 
          "-ir" + _numIRDocs + "-plates" + _numTemplates + "-topics" + _numTopics + "-jp" + _numJunkTemplates + "-jt" + _numJunkTopics + ".model");
    }
    return sampler;
  }

  /**
   * This function runs inference, but it does so many times using different probability
   * cutoffs. It prints the final F1 score for each parameter combination at the very end.
   *  
   * This should be changed for testing to just run one of them: the best one on the dev set.
   */
  public void runInference() {
    System.out.println("Time to infer!");
    Inference infer = new Inference(_modelPath, _inferMaxEntities, _inferMinProb);
    final int[] maxPerRoles = { 1, 2, 3, 4 };
    final double[] probs = { .9, .8, .7, .6, .5, .4, .3 };
//    final int[] maxPerRoles = {  3 };
//    final double[] probs = { .98 };
    double[][][] prf1s = new double[maxPerRoles.length][][];
    int ii = 0;

    // Load the answer key for MUC or Corporate Acq.
    KeyReader answerKey = getAnswerKey();
    
    // Set the appropriate test set (train set often is the test set during development)
    List<List<TextEntity>> testEntities = _trainDocsEntities;
    List<String> testDocnames = _trainDocsNames;
    if( _testDocsEntities != null ) testEntities = _testDocsEntities;
    if( _testDocsEntities != null ) testDocnames = _testDocsNames;
    
    EvaluateModel evaluator = new EvaluateModel(infer.sampler.numTopics, answerKey);
    evaluator._debugOn = _debugOn;

    // Try different thresholds for the max per role map and the minimum acceptable probability.
    for( int maxPerRole : maxPerRoles ) {
      prf1s[ii] = new double[probs.length][];
      int jj = 0;
      for( double minProb : probs ) {
        System.out.println("--> minProb = " + minProb);
        System.out.println("--> maxPerSlot = " + maxPerRole);
        
        // Reset the labels on entities.
        infer._minAcceptableProbability = minProb;
        infer._maxEntitiesPerRole = maxPerRole;
        Inference.clearEntityLabels(testEntities);

        // Label entities with slots based on their already calculated probabilities.
        int doci = 0;
        for( List<TextEntity> doc : testEntities ) {
          //        System.out.println("** Inference Doc " + ii + " " + _docsNames.get(ii++) + " **");

          // Don't label a document if its average entity likelihood is too low.
//          double likelihood = infer.computeDocLikelihood(doc);
//          System.out.println("** Inference Likelihood doc " + doci + "\t" + _docsNames.get(doci) + "\t" + likelihood +
//              "\t" + (likelihood/doc.size()));
//          if( (likelihood/doc.size()) > -13.0 )
          
          System.out.println("label entities doc " + testDocnames.get(doci));
          infer.labelEntities(doc, _debugOn);
          doci++;
        }

        // Save our guesses to the evaluator.
        evaluator.setGuesses(testDocnames, testEntities);

        // Find the best topic/slot mapping.
        System.out.println("** Evaluate Entities **");
        double[] avgPRF1 = evaluator.evaluateSlotsGreedy(_inferMaxRolesPerSlot);
        
        // Save this for printing out later.
        prf1s[ii][jj++] = avgPRF1;
      }
      ii++;
    }
    
    // Print results of all combinations.
    ii = 0;
    for( double[][] doub : prf1s ) {
      int jj = 0;
      for( double[] prf1 : doub )
        System.out.printf("m=%d,p=%.1f\tall\tprec=%.3f\trecall=%.3f\tf1=%.2f\n", maxPerRoles[ii], probs[jj++], prf1[0], prf1[1], prf1[2]);
      ii++;
    }
  }
  
  /**
   * @param args
   */
  public static void main(String[] args) {
    Learner learner = new Learner(args);
    
    if( learner._modelPath != null ) {
      if( learner._inferBySampler )
      	learner.inferUsingSampledData(null);
      else
      	learner.runInference();
    }
    else if( learner._learnAndInfer )
//      learner.learnAndInferAvg();
    	learner.learnAndInferSampledAvg();
    else
      learner.learn();
  }

}
