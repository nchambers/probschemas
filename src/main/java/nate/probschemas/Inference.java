package nate.probschemas;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import nate.util.Util;

import edu.stanford.nlp.math.ArrayMath;

/**
 * This class uses a pre-trained graphical model to label given documents.
 * This uses the entity representation of documents, accepting a list of TextEntity objects
 * as a single document.
 * We label using the learned conditional probability tables from the Gibbs Sampler. 
 * 
 *
 */
public class Inference {
  public GibbsSamplerEntities sampler;
  double _minAcceptableProbability = 0.95;
  int _maxEntitiesPerRole = 3; // cannot assign more than this number of entities to any single role
  boolean _oneLabelPerEntity = true; // if false, more than one topic can label an entity
  boolean _skipPoorDocuments = false; // if true, a topic does not label documents if its top verbs doesn't appear
  
  public Inference(String modelPath) {
    init(modelPath);
  }
  
  public Inference(String modelPath, int maxEntities, double minProb) {
    _minAcceptableProbability = minProb;
    _maxEntitiesPerRole = maxEntities;
    init(modelPath);
  }
  
  public Inference(GibbsSamplerEntities sampler, int maxEntities, double minProb) {
    _minAcceptableProbability = minProb;
    _maxEntitiesPerRole = maxEntities;
    this.sampler = sampler;
  }

  private void init(String modelPath) {
    this.sampler = GibbsSamplerEntities.fromFile(modelPath);
    System.out.println("Loaded model from " + modelPath);
    System.out.println("Loaded model contains " + this.sampler.numTopics + " topics");
    System.out.println("Minimum acceptable probability: " + _minAcceptableProbability);
    System.out.println("Max entities per role: " + _maxEntitiesPerRole);
    this.sampler.printWordDistributionsPerTopic();
    System.out.println("\n");
  }
  
  /**
   * Grab all of the predicates from all of the mentions.
   * @param entity A single entity.
   * @return A set of predicates from the entity's mentions.
   */
  private Set<String> getVerbsInEntity(TextEntity entity) {
  	Set<String> words = new HashSet<String>();
  	for( int ii = 0; ii < entity.numMentions(); ii++ ) {
      String dep = entity.getMentionDependency(ii);
      String predicate = dep.substring(dep.indexOf("--")+2);
      words.add(predicate);
    }
  	return words;
  }
  
  /**
   * Grab all of the predicates from all of the mentions in all entities.
   * @param docEntities The document's entities.
   * @return A set of predicates from the entities' mentions.
   */
  private Set<String> getVerbsInDocFromDeps(List<TextEntity> docEntities) {
  	Set<String> words = new HashSet<String>();
    for( TextEntity entity : docEntities )
    	words.addAll(getVerbsInEntity(entity));
    return words;
  }
  
  /**
   * Get the 4 best predicates (e.g., kidnap) from the current model.
   * @return
   */
  private Set<String> getTopPredicatesBasedOnDeps() {
    List<String> orderedPredicates = sampler.getTopPredicatesBasedOnDeps();
    Set<String> preds = new HashSet<String>();
    for( int xx = 0; xx < 4 && xx < orderedPredicates.size(); xx++ )
      preds.add(orderedPredicates.get(xx));
    return preds;
  }
  
  /**
   * Get the top n verbs in each topic. 
   * @return An array the length of the number of topics. Each cell is the top n verbs for that topic.
   */
  private List<String>[] topVerbsByTopic() {
    List<String>[] bytopic = new ArrayList[sampler.numTopics];
    int n = 2;
    
    for( int topicid = 0; topicid < sampler.numTopics; topicid++ )
      bytopic[topicid] = sampler.getTopVerbsInTopic(topicid, n, 0.05);

    return bytopic;
  }
  
  private boolean predicateOverlap(List<String> preds, Set<String> preds2) {
  	for( int xx = 0; xx < 5 && xx < preds.size(); xx++ )
  		if( preds2.contains(preds.get(xx)) )
  			return true;
  	return false;  	
  }
  
  /**
   * Assumes the entities are labeled with topics/slots.
   */
  public double computeDocLikelihood(final List<TextEntity> docEntities) {
    double likelihood = 0.0;
    for( TextEntity entity : docEntities ) {
      // Probs are set in log-space.
      double[] probs = new double[sampler.numTopics];
      getTopicDistribution(entity, probs);
      likelihood += ArrayMath.max(probs);
    }
    return likelihood;
  }

  
  /**
   * Remove any slot labels that have been assigned to entities. Clear the board.
   */
  public static void clearEntityLabels(List<List<TextEntity>> docsEntities) {
    for( List<TextEntity> docEntities : docsEntities ) {
      for( TextEntity entity : docEntities )
        entity.clearLabels();
    }
  }

  /**
   * Calculate the topic distribution for a single entity. Fills the given topic prob array.
   * Probs are in log space.
   */
  private void getTopicDistribution(final TextEntity entity, double[] probs) {
    for( int topic = 0; topic < sampler.numTopics; topic++ ) {
      // P( z ) 
      double probOfTopic = sampler.probOfTopic(topic);
      // P( w | slot )
      double probOfWGivenTopic = sampler.probOfWGivenTopic(entity.getCoreToken(), topic);
      probs[topic] = Math.log(probOfTopic * probOfWGivenTopic);
      
      if( sampler.includeEntityFeatures ) {
        double prob = sampler.probOfFeatsGivenTopic(entity.types, topic);
//        System.out.println("feats p=" + prob);
        probs[topic] += Math.log(prob);
      }
      
      for( int mention = 0; mention < entity.numMentions(); mention++ ) {
      	String depStr = entity.deps.get(mention);
      	
      	// P( dep | slot )
      	double prob = sampler.probOfDepGivenTopic(depStr, topic);
      	probs[topic] += Math.log(prob);

      	// P( verb | slot )
      	if( sampler.includeVerbs ) {
      		String verb = depStr.substring(depStr.indexOf("--")+2);
      		prob = sampler.probOfVerbGivenTopic(verb, topic);
      		probs[topic] += Math.log(prob);
//      		System.out.println("v=" + verb + "\tp=" + prob);
      	}
      }
    }
    ArrayMath.logNormalize(probs);
  }
  
  /**
   * This function computes P(role | entity) for all entities. It does so by computing the 
   * joint P(role,entity) using the graphical model, and then normalizing by Sum_r P(r,entity).
   * This is easy to do since the role assignments are independent of each other in the model.
   * 
   * The role labels are assigned to each entity destructively, setting the label variable in 
   * each TextEntity object.
   * 
   * @param docEntities The list of entities to infer labels.
   */
  public void labelEntities(List<TextEntity> docEntities, boolean debug) {
    // Store the best entities scored with each role.
    Map<Integer,Double>[] entityToProbability = new HashMap[sampler.numTopics];
    for( int topic = 0; topic < sampler.numTopics; topic++ )
      entityToProbability[topic] = new HashMap<Integer,Double>();
    
    // DEBUG FOR NOW
    Set<String> predicatesInDoc = getVerbsInDocFromDeps(docEntities);
    Set<String> learnedTopPredicates = getTopPredicatesBasedOnDeps();
    System.out.println("Learned top predicates: " + learnedTopPredicates);
    System.out.println("Doc predicate words: " + predicatesInDoc);
//    sampler.printWordDistributionsPerTopic();
    boolean skipdoc = false;
//    boolean skipdoc = true;
//    for( String pred : learnedTopPredicates ) {
//      if( predicatesInDoc.contains(pred) ) {
//        skipdoc = false;
//      }
//    }
//    if( skipdoc ) System.out.println("Skipping doc!");
    List<String>[] topverbs = topVerbsByTopic();
//    for( int tt = 0; tt < topverbs.length; tt++ )
//      System.out.println("topic " + tt + ":\t" + topverbs[tt]);
    
    
    if( !skipdoc ) {
      // Run inference over each entity and keep the best scoring role. 
      int entityid = 0;
      for( TextEntity entity : docEntities ) {
        double best = -Double.MAX_VALUE;
        int bestTopic = -1;
	//        Set<String> entityPredicates = getVerbsInEntity(entity); 
        double[] probs = new double[sampler.numTopics];
        getTopicDistribution(entity, probs);

        // Find the most probable topic for this entity.
        for( int topic = 0; topic < sampler.numTopics; topic++ ) {
        	// if( !predicateOverlap(topverbs[topic], entityPredicates) ) {
        	// 	probs[topic] = 0;
        	// 	for(int xx = 0; xx < 5 && xx < topverbs[topic].size(); xx++ )
        	// 		System.out.print(topverbs[topic].get(xx) + " ");
        	// 	System.out.println("\n\t----" + entityPredicates);
        	// }
        	
          if( probs[topic] > best ) {
            best = probs[topic];
            bestTopic = topic;
          }
        }

        // **********************************
        // Debugging.
        if( debug ) {
          System.out.println("Entity: " + entity);
          System.out.print("\tprobs:\t\t");
          for( int topic = 0; topic < sampler.numTopics; topic++ ) {
            if( topic == bestTopic )
              System.out.print("*");
            System.out.printf("%.2f ", probs[topic]);
          }
          System.out.println();
        }
        if( debug ) {
          System.out.print("\tnormed probs:\t");
          for( int topic = 0; topic < sampler.numTopics; topic++ ) {
            if( topic == bestTopic )
              System.out.print("*");
            System.out.printf("%.6f ", Math.exp(probs[topic]));
          }
          System.out.println();
        }
        // **********************************

        // Save the entity probabilities per role to be sorted next.
        if( debug ) System.out.printf("Checking threshold against %.4f\n", Math.exp(probs[bestTopic]));
        entity.clearLabels(); // default
        if( Math.exp(probs[bestTopic]) > _minAcceptableProbability ) {
          entityToProbability[bestTopic].put(entityid, Math.exp(probs[bestTopic]));
          if( debug ) System.out.println("  - saved " + bestTopic);
        }
        entityid++;
      }

      // Label the top n entities for each role.
      for( int topic = 0; topic < sampler.numTopics; topic++ ) {
        
        // Check if this topic applies to this document.
        skipdoc = false;
        if( _skipPoorDocuments ) {
        	skipdoc = true;
        	for( String pred : predicatesInDoc ) {
        		if( topverbs[topic].contains(pred) )
        			skipdoc = false;
        	}
        	if( skipdoc ) System.out.println("Skipped topic " + topic);
        }
        
        // Label the entity for real now.
        if( !skipdoc ) {
          List<Integer> topentities = Util.sortKeysByValues(entityToProbability[topic]);
          for( int xx = 0; xx < _maxEntitiesPerRole && xx < topentities.size(); xx++ ) {
            docEntities.get(topentities.get(xx)).addLabel(topic);
            if( debug ) System.out.println("Labeled entity " + docEntities.get(topentities.get(xx)) + " with " + topic);
          }
        } else {
          
        }
      }
      if( debug ) System.out.println("\n");
    }
  }

}
