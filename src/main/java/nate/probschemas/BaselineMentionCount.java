package nate.probschemas;

import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import nate.util.Pair;
import nate.muc.MUCKeyReader;
import nate.ProcessedData;
import nate.util.Directory;
import nate.util.HandleParameters;

/**
 * This is a baseline for template extraction based solely on how many times each entity
 * in a document is repeated. It uses the number of entity mentions per entity, and assigns
 * the most frequent entity to the first template slot. The 2nd most frequent to the second
 * slot, and so on. Every document assigns every slot. The "-m" flag lets you assign more
 * than one entity per slot, but this doesn't change the results on MUC much.
 * 
 * There are 4 slots (hardcoded size 4) in this baseline. 
 * 
 * -m <int>
 * The number of entities to assign to each template slot.
 * 
 * -key <path>
 * The gold template file.
 * 
 */
public class BaselineMentionCount {
  String _dataDir = null;
  String _mucKeyPath = "/home/nchamber/corpora/muc34/TASK/CORPORA/key-kidnap.muc4";
  List<List<TextEntity>> _docsEntities;
  List<String> _docsNames;
  
  int _numSlots = 4;   // the number of slots we give to every template
  int _numPerSlot = 1; // the number of entities put into each slot
  final String[] _types = { "KIDNAP", "BOMBING", "ATTACK", "FORCED WORK STOPPAGE", "ROBBERY", "ARSON" };

  
  public BaselineMentionCount(String[] args) {
    HandleParameters params = new HandleParameters(args);
    
    if( args.length < 1 ) {
      System.out.println("BaselineMentionCount -key <gold-templates> [-m <int>] <data-dir>");
      System.exit(-1);
    }

    if( params.hasFlag("-key") )  _mucKeyPath = params.get("-key");
    if( params.hasFlag("-m") )  _numPerSlot = Integer.parseInt(params.get("-m"));
    
    System.out.println("Num slots:\t" + _numSlots);
    System.out.println("Num per slot:\t" + _numPerSlot);
    System.out.println("Gold Templates:\t" + _mucKeyPath);
    
    // Load the given training data.
    _dataDir = args[args.length-1];
    load(_dataDir);
  }
  
  /**
   * Populates the _docsEntities variable based on this data directory.
   * @param dataDir Path to a directory that contains the four key files.
   * @param n The first n documents are read. 
   */
  public void load(String dataDir) {
    String parsesPath = dataDir + File.separator + Directory.nearestFile("parse", dataDir);
    String depsPath   = dataDir + File.separator + Directory.nearestFile("deps", dataDir);
    String eventsPath = dataDir + File.separator + Directory.nearestFile("events", dataDir);
    String nerPath    = dataDir + File.separator + Directory.nearestFile("ner", dataDir);
    System.out.println("parses: " + parsesPath + " and deps: " + depsPath + " and events: " + eventsPath + "and ner: " + nerPath);
    
    // Initialize all forms of this learner, but really only _docsEntities is used I think.
    _docsEntities = new ArrayList<List<TextEntity>>();
    _docsNames = new ArrayList<String>();
    

    DataSimplifier simplify = new DataSimplifier(0,0);
    
    // Check the cache of this file. Read from there if we already processed it!
    Pair<List<String>,List<List<TextEntity>>> cached = simplify.getCachedEntityList(parsesPath);
    if( cached != null ) {
      _docsEntities = cached.second();
      _docsNames = cached.first();
      System.out.println("Cached number of docs in entity list: " + _docsEntities.size());
      System.out.println("Cached doc names: " + _docsNames);
    }
    else {
      // Read the data files from disk.
      ProcessedData loadedData = new ProcessedData(parsesPath, depsPath, eventsPath, nerPath);
      loadedData.nextStory();

      // Get the tokens and dependencies from this file.
      _docsEntities = simplify.getEntityList(loadedData, _docsNames, Integer.MAX_VALUE);

      // Write to cache.
      simplify.writeToCache(parsesPath, _docsNames, _docsEntities);
    }
  }

  /**
   * Return a random entity from the given list of entities.
   */
  private TextEntity getRandom(List<TextEntity> entities) {
    if( entities == null ) return null;
    if( entities.size() == 1 ) return entities.get(0);
    
    Random rand = new Random();
    return entities.get(rand.nextInt(entities.size()));
  }
  
  /**
   * This function destructively labels each given TextEntity with a slot number.
   * Some entities are not labeled.
   * @param docEntities The list of entities from a single document.
   */
  private void labelEntities(List<TextEntity> docEntities) {
    int maxlength = 0;
    for( TextEntity entity : docEntities )
      if( entity.numMentions() > maxlength )
        maxlength = entity.numMentions();
    
    // The 3rd cell in this array will contain all entities with 3 mentions.
    // The 2nd cell all entities with 2 mentions.
    // etc.
    List<TextEntity>[] entitiesByLength = new ArrayList[maxlength+1];
    for( int ii = 0; ii < maxlength+1; ii++ )
      entitiesByLength[ii] = new ArrayList<TextEntity>();
    for( TextEntity entity : docEntities )
      entitiesByLength[entity.numMentions()].add(entity);
    
    // Add the entities with the most mentions.
    int chosenSlot = 0;
    int addedPerSlot = 0;
    int ii = maxlength-1;
    while( ii > 0 ) {
      List<TextEntity> entities = entitiesByLength[ii];
      if( entities != null ) {
        while( entities.size() > 0 && chosenSlot < _numSlots ) {
          TextEntity chosenEntity = getRandom(entities);
          chosenEntity.addLabel(chosenSlot);
          addedPerSlot++;
          if( addedPerSlot == _numPerSlot ) {
            addedPerSlot = 0;
            chosenSlot++;
          }
          entities.remove(chosenEntity);
        }
      }
      ii--;
    }
    
    for( TextEntity entity : docEntities )
      if( entity.hasALabel() ) System.out.println("** " + entity.getLabels() + ":\t" + entity);
    System.out.println("*********************** DONE LABELING ********************");
  }
  
  /**
   * Label each entity with a slot (or no slot). Give it to the evaluator and print the results.
   */
  public void inferAndEvaluate() {
    // Create the Evaluation Object.
    MUCKeyReader answerKey = new MUCKeyReader(_mucKeyPath);
    EvaluateModel evaluator = new EvaluateModel(_numSlots, answerKey);
    evaluator._debugOn = false;
    
    // Label entities with slots based on their already calculated probabilities.
    Inference.clearEntityLabels(_docsEntities);
    for( List<TextEntity> doc : _docsEntities ) {
      //        System.out.println("** Inference Doc " + ii + " " + _docsNames.get(ii++) + " **");
      labelEntities(doc);
    }

    // Save our guesses to the evaluator.
    evaluator.setGuesses(_docsNames, _docsEntities);

    // Find the best topic/slot mapping.
    System.out.println("** Evaluate Entities **");
    double[] avgPRF1 = evaluator.evaluateSlotsGreedy(Integer.MAX_VALUE);
  }
  
  
  public static void main(String[] args) {
    BaselineMentionCount base = new BaselineMentionCount(args);
    base.inferAndEvaluate();
  }

}
