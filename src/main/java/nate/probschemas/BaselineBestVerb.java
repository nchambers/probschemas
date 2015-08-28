package nate.probschemas;

import java.io.File;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;

import nate.CalculateIDF;
import nate.CountVerbDepCorefs;
import nate.IDFMap;
import nate.util.Pair;
import nate.WordEvent;
import nate.muc.MUCKeyReader;
import nate.ProcessedData;
import nate.util.Directory;
import nate.util.HandleParameters;
import nate.util.Util;
import nate.util.WordNet;

/**
 * This class depends on IDF counts of words from a general corpus and the domain corpus.
 * It looks for a .idf file in the domain's directory, so precalculate it and store it there.
 * 
 * This baseline uses Filatova's approach of comparing a verb's frequency in the domain to
 * the frequency in a general corpus. It calculates conditional probability of a word, multiplied 
 * by an IDF-like score. After given the top verbs, I label the subject and object of each verb
 * with a different template slot ID. So each template slot has one and only one verb/dep pattern. 
 * 
 * BaselineBestVerb -train <dir> -test <dir> -key <path> [-idf <path>] [-n <int>]
 *  
 * 
 * -n <int>
 * The number of slots to extract, we align the best during test.
 * A higher number sort of cheats, but we do that for LDA all the time...
 * 
 * -train <dir>
 * The directory containing parsed files of training data.
 * 
 * -test <dir>
 * The directory containing parsed files of testing data.
 * 
 * -key <path>
 * The gold templates for the test docs.
 * 
 * -idf <path>
 * General IDF counts from gigaword. (if not given, searches for it anyway)
 * 
 * -m <int>
 * The number of training docs to use from the given training set. If given, we randomly
 * choose m docs and run the baseline. We do this 20 times and average the result.
 * 
 */
public class BaselineBestVerb {
  String _trainDir = null;
  String _testDir = null;
  String _mucKeyPath = "/home/nchamber/corpora/muc34/TASK/CORPORA/key-kidnap.muc4";
//  List<List<TextEntity>> _docsEntitiesTrain;
  List<List<TextEntity>> _docsEntitiesTest;
  List<String> _docsNamesTrain;
  List<String> _docsNamesTest;
  
  IDFMap generalIDF;
  IDFMap trainingIDF;
  CountVerbDepCorefs trainingPatterns;
  
  ProcessedData trainData;
  
  int _numTrainingDocs = Integer.MAX_VALUE;
  int _numSlots = 5;   // the number of slots we give to every template
  final String[] _types = { "KIDNAP", "BOMBING", "ATTACK", "FORCED WORK STOPPAGE", "ROBBERY", "ARSON" };
  
  // Verb/dep -> slot number
  List<Pair<String,Integer>> _slotPatterns;

  
  public BaselineBestVerb(String[] args) {
    HandleParameters params = new HandleParameters(args);
    
    if( args.length < 1 ) {
      System.out.println("BaselineMentionCount [-idf <giga-path>] [-n <int>] -key <gold-templates> -train <dir> -test <dir>");
      System.exit(-1);
    }

    if( params.hasFlag("-key") )   _mucKeyPath = params.get("-key");
    if( params.hasFlag("-n") )     _numSlots = Integer.parseInt(params.get("-n"));
    if( params.hasFlag("-m") )     _numTrainingDocs = Integer.parseInt(params.get("-m"));
    if( params.hasFlag("-train") ) _trainDir = params.get("-train");
    if( params.hasFlag("-test") )  _testDir = params.get("-test");
    
    System.out.println("Num slots:\t" + _numSlots);
    System.out.println("Gold Templates:\t" + _mucKeyPath);
    System.out.println("Training:\t" + _trainDir);
    System.out.println("Testing:\t" + _testDir);
    
    // Read the verb/arg patterns and IDF counts.
    if( params.hasFlag("-idf") ) generalIDF = new IDFMap(params.get("-idf"));
    else generalIDF = new IDFMap(IDFMap.findIDFPath());
//    trainingIDF = new IDFMap(_trainDir + File.separator + Directory.nearestFile("tokens-lemmas.idf", _trainDir));
    
    // Load the domain train/test sets.
    load(_trainDir, false);
    load(_testDir, true);
  }
  
  /**
   * Populates the _docsEntities variable based on this data directory.
   * @param dataDir Path to a directory that contains the four key files.
   * @param n The first n documents are read. 
   */
  public void load(String dataDir, boolean test) {
    String parsesPath = dataDir + File.separator + Directory.nearestFile("parse", dataDir);
    String depsPath   = dataDir + File.separator + Directory.nearestFile("deps", dataDir);
    String eventsPath = dataDir + File.separator + Directory.nearestFile("events", dataDir);
    String nerPath    = dataDir + File.separator + Directory.nearestFile("ner", dataDir);
    System.out.println("parses: " + parsesPath + " and deps: " + depsPath + " and events: " + eventsPath + "and ner: " + nerPath);
    
    // Initialize all forms of this learner, but really only _docsEntities is used I think.
    List<List<TextEntity>> docsEntities = new ArrayList<List<TextEntity>>();
    List<String> docsNames = new ArrayList<String>();

    DataSimplifier simplify = new DataSimplifier();
    
    // Check the cache of this file. Read from there if we already processed it!
    Pair<List<String>,List<List<TextEntity>>> cached = simplify.getCachedEntityList(parsesPath);
    if( cached != null ) {
      docsEntities = cached.second();
      docsNames = cached.first();
      System.out.println("Cached number of docs in entity list: " + docsEntities.size());
      System.out.println("Cached doc names: " + docsNames);
    }
    else {
      // Read the data files from disk.
      ProcessedData loadedData = new ProcessedData(parsesPath, depsPath, eventsPath, nerPath);
      loadedData.nextStory();

      // Get the tokens and dependencies from this file.
      docsEntities = simplify.getEntityList(loadedData, docsNames, Integer.MAX_VALUE);

      // Write to cache.
      simplify.writeToCache(parsesPath, docsNames, docsEntities);
    }
    
    if( test ) {
      _docsEntitiesTest = docsEntities;
      _docsNamesTest = docsNames;
    } else {
      _docsNamesTrain = docsNames;
      trainData = new ProcessedData(parsesPath,null,null,null);
      trainingIDF = computeIDF(trainData, null, null);
    }
  }
  
  /**
   * Calculate word IDF scores on the fly, only on the given training data.
   * @param data The documents we count tokens in.
   * @param docnames The stories we count. All others are ignored. If null, count all stories.
   */
  public static IDFMap computeIDF(ProcessedData data, List<String> docnames, WordNet wordnet) {
  	CalculateIDF calc;
  	
  	if( wordnet != null )
  		calc = new CalculateIDF(wordnet);
  	else {
  		String[] args = { "-wordnet", WordNet.findWordnetPath() };
  		calc = new CalculateIDF(args);
  	}
        
    data.reset();
    data.nextStory();
    List<String> strs = data.getParseStrings();
    while( strs != null ) {
      if( docnames == null || docnames.contains(data.currentStory()) )
        calc.countStory(strs);
      data.nextStory();
      strs = data.getParseStrings();
    }
    
    System.out.println("Kidnap IDF: " + calc._idfLemmas.getFrequency("v-kidnap") + "\t" + calc._idfLemmas.getDocCount("v-kidnap"));
    calc.calculateIDF();
    return calc._idfLemmas;
  }
  
  /**
   * Adapted from my code in: TemplateExtractor.java
   * 
   * This is an implementation of Filatova's 2006 paper.
   * Finds syntactic relations that have high tf*idf scores where the tf
   * is from the desired domain, and the idf is from a general corpus.
   * @param verbLookupType Determines how we look up the key verbs: fila, likelihood, salience, chambers
   * @param scoreRelationWithArg If true, then domain patterns are found with their relations: kidnap-s:terrorist
   *                             If false, then patterns are just scored as: kidnap-s
   *                             Filatova did the first one...found the most frequent pattern+arg.
   * @param resolveEntities If true, seen args are replaced with their entity class' best string.
   *                        Filatova didn't do this, just used the base arg.                            
   */
  private void computeLikelihoods() {
    // Get the most unique words for this domain - Filatova approach.
    DomainVerbDetector verbDetector = new DomainVerbDetector(trainingIDF, generalIDF);
    List<String> topWords = verbDetector.detectWordsFilatova(true);

    // Filatova's paper takes the top 50.
    Util.firstN(topWords, 50);

    for( String word : topWords )
      System.out.println("top: " + word + " idf=" + generalIDF.get(word));

    // Take the top verbs and mark their subjects/objects with slot IDs.
    _slotPatterns = new ArrayList<Pair<String,Integer>>();
    for( int slotid = 0; slotid < _numSlots; slotid++ ) {
      if( topWords.size() < (slotid/2+1) )
        break;
      if( slotid % 2 == 0 ) {
        _slotPatterns.add(new Pair<String,Integer>(topWords.get(slotid/2) + ":" + WordEvent.DEP_SUBJECT, slotid));
        System.out.println(topWords.get(slotid/2) + ":" + WordEvent.DEP_SUBJECT + "\t" + slotid);
      }
      else {
        _slotPatterns.add(new Pair<String,Integer>(topWords.get(slotid/2) + ":" + WordEvent.DEP_OBJECT, slotid));
        System.out.println(topWords.get(slotid/2) + ":" + WordEvent.DEP_OBJECT + "\t" + slotid);
      }
    }    
  }
  
  /**
   * Use the global test set entity list per doc, and pull out a random number of documents (num docs)
   * that will be used for learning. Put those num documents into the given fillEntities list.
   * @param num The number of random documents to select.
   * @param fillNames The list of document names selected.
   * @param fillEntities The entity lists for each document selected.
   */
  private List<String> getRandomTrainingDocs(int num) {
    List<String> fillNames = new ArrayList<String>();
    
    // Sanity check. We can't add more docs than there are in existence.
    if( num > _docsNamesTrain.size() )
      num = _docsNamesTrain.size();
    
    Random rand = new Random(); 
    Set<Integer> docids = new HashSet<Integer>();
    while( docids.size() < num )
      docids.add(rand.nextInt(_docsNamesTrain.size()));
    
    for( int ii = 0; ii < _docsNamesTrain.size(); ii++ ) {
      if( docids.contains(ii) ) {
        fillNames.add(_docsNamesTrain.get(ii));
      }
    }
    return fillNames;
  }
  
  /**
   * Label an entity with a template slot if it contains a mention inside one of the top verb/dep slots.
   * If the entity has multiple mentions that match, choose the lowest numbered slot (the most frequently seen one). 
   * @param docEntities A single doc's entities.
   */
  private void labelEntities(List<TextEntity> docEntities) {
    for( TextEntity entity : docEntities ) {
      int bestmatch = Integer.MAX_VALUE;
      int bestslot = -1;
      
      for( int ii = 0; ii < entity.numMentions(); ii++ ) {
        // Just guess that it is a verb, if it is not, who cares, we only have verbs as the top slots so it won't match anyway.
        String mentionslot = entity.getMentionDependency(ii); // dobj--kidnap
        String verb = mentionslot.substring(mentionslot.indexOf("--")+2); // kidnap
        String dep = mentionslot.substring(0, mentionslot.indexOf("--")); // dobj
        String slot = "v-" + verb + ":" + WordEvent.normalizeRelation(dep); // v-kidnap:o
//        System.out.println("slot created: " + slot + "\tfrom " + entity.getMentionToken(ii) + " dep " + entity.getMentionDependency(ii));
        int patterni = 0;
        for( Pair<String,Integer> pattern : _slotPatterns ) {
          if( slot.equals(pattern.first()) ) {
            if( patterni < bestmatch ) {
              bestmatch = patterni;
              bestslot = pattern.second();
            }
          }
        }
      }
      
      if( bestslot > -1 ) {
        entity.addLabel(bestslot);
        System.out.println("Labeled: " + bestslot + "\t" + bestmatch + "\t" + entity);
      }
    }
  }

  /**
   * Train the baseline over a subset of the training data, and do it 20 times, averaging the
   * results on the entire test set. The training data just computes the IDF scores, so we do
   * this 20 times and run the 20 different counts on the test set.
   */
  public void averageRuns() {
    System.out.println("Average Runs!");
    MUCKeyReader answerKey = new MUCKeyReader(_mucKeyPath);
    EvaluateModel evaluator = new EvaluateModel(_numSlots, answerKey);
    WordNet wordnet = new WordNet(WordNet.findWordnetPath());
    int numRuns = 20, runi = 0;;
    double[][] allruns = new double[numRuns][];
                                    
    double[] sumPRF1 = { 0, 0, 0 };
      
    for( int run = 0; run < numRuns; run++ ) {
      System.out.println("Learn and Infer run " + run);

      // Get a random subset of the data.
      List<String> idocsNames = getRandomTrainingDocs(_numTrainingDocs);
      System.out.println("rand docs: " + idocsNames);

      // Compute IDF scores for this subset of training.
      trainingIDF = computeIDF(trainData, idocsNames, wordnet);
      computeLikelihoods();

      // Label the test set entities with slots based on the IDF scores.
      Inference.clearEntityLabels(_docsEntitiesTest);
      for( List<TextEntity> doc : _docsEntitiesTest )
        labelEntities(doc);
      // Save our guesses to the evaluator.
      evaluator.setGuesses(_docsNamesTest, _docsEntitiesTest);
      System.out.println("Set guesses on " + _docsNamesTest.size() + " docs and " + _docsEntitiesTest.size() + " entities.");

      // Find the best topic/slot mapping.
      System.out.println("** Evaluate Entities **");
      double[] avgPRF1 = evaluator.evaluateSlotsGreedy(Integer.MAX_VALUE);
      allruns[runi++] = avgPRF1;
      for( int ii = 0; ii < sumPRF1.length; ii++ )
        sumPRF1[ii] += avgPRF1[ii];
    }

    // Average.
    for( int ii = 0; ii < sumPRF1.length; ii++ )
      sumPRF1[ii] /= numRuns;
    for( int ii = 0; ii < allruns.length; ii++ )
      System.out.printf("run %d:\tp=%.3f\tr=%.3f\tf1=%.2f\n", ii, allruns[ii][0], allruns[ii][1], allruns[ii][2]);
    System.out.printf("Average of " + numRuns + " runs:\tp=%.3f\tr=%.3f\tf1=%.2f\n", sumPRF1[0], sumPRF1[1], sumPRF1[2]);
  }
  
  /**
   * Label each entity with a slot (or no slot). Give it to the evaluator and print the results.
   */
  public void inferAndEvaluate() {
    computeLikelihoods();
    
    // Create the Evaluation Object.
    MUCKeyReader answerKey = new MUCKeyReader(_mucKeyPath);
    EvaluateModel evaluator = new EvaluateModel(_numSlots, answerKey);
    evaluator._debugOn = false;
    
    // Label entities with slots based on their already calculated probabilities.
    Inference.clearEntityLabels(_docsEntitiesTest);
    for( List<TextEntity> doc : _docsEntitiesTest ) {
      //        System.out.println("** Inference Doc " + ii + " " + _docsNames.get(ii++) + " **");
      labelEntities(doc);
    }

    // Save our guesses to the evaluator.
    evaluator.setGuesses(_docsNamesTest, _docsEntitiesTest);

    // Find the best topic/slot mapping.
    System.out.println("** Evaluate Entities **");
    evaluator.evaluateSlotsGreedy(Integer.MAX_VALUE);
  }
  
  public void runit() {
    if( _numTrainingDocs < Integer.MAX_VALUE )
      averageRuns();
    else
      inferAndEvaluate();
  }
  
  /**
   * @param args
   */
  public static void main(String[] args) {
    BaselineBestVerb base = new BaselineBestVerb(args);
    base.runit();
  }

}
