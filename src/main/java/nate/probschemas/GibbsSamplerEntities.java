package nate.probschemas;

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

import nate.util.Pair;
import nate.util.Util;

import edu.stanford.nlp.math.ArrayMath;
import edu.stanford.nlp.math.SloppyMath;
import edu.stanford.nlp.stats.ClassicCounter;
import edu.stanford.nlp.stats.Counter;
import edu.stanford.nlp.util.ErasureUtils;
import edu.stanford.nlp.util.HashIndex;
import edu.stanford.nlp.util.Index;

/**
 * Runs a gibbs sampler on the following graphical model: z -> e -> || w,d ||
 * Every entity receives a topic/role label, and each entity is made up of
 * entity mentions, so it is determined by: P(z) * Product_i( P(wi|z)P(di|z) )
 * 
 * It also runs on a more complicated model from Chambers 2013. t -> z -> e -> || w,d || 
 *                                                                -> || v ||
 *
 * "templates" - Number of higher-level scenarios, e.g., kidnap or arrest
 * "topics" - Number of overall roles across all templates, e.g., the "location of an arrest"
 *          - Topics must be divisible by the number of templates (4 templates and 4 roles in each = 16 topics)
 *          
 */
public class GibbsSamplerEntities implements Sampler, Serializable {
  static final long serialVersionUID = 10000;

  public Index<String> wordIndex;
  public Index<String> verbIndex;
  public Index<String> depIndex;
  private final Random random;
  public List<String> docNames;

  public boolean thetasInDoc = true;
  public boolean includeVerbs = false;
  public boolean includeEntityFeatures = true;
  public boolean constrainInverseDeps = true;
  int maxEntitiesPerTopic = 2; // (constrained sampler mode) each document can only map this many entities to a single role

  // Model.
  private int[][][] words; // [doc][entity][mention]
  private int[][][] deps;  // [doc][entity][mention]
  private int[][][] verbs; // [doc][entity][mention]
  private int[][][] inverseDeps;  // [doc][entity][mention]   : the inverse of "subj" is "obj" and vice versa
  private int[][][] feats;  // [doc][entity][feattype]
  private int[][] zs;      // [doc][entity]  zs are z variable assignments to single entities
  private double[] topicCounts;      // when one global Theta distribution
  private int[][] topicCountsByDoc;  // when Thetas are per document
  private int numEntitiesInAllDocs = 0; // # of entities in dataset
  private int numMentionsInAllDocs = 0; // # of entity mentions in dataset
  private ClassicCounter<Integer>[] wCountsBySlot;
  private ClassicCounter<Integer>[] verbCountsBySlot;
  private ClassicCounter<Integer>[] depCountsBySlot;
  private ClassicCounter<Integer>[] featCountsBySlot;
  public final int numFeats = TextEntity.TYPE.values().length;

  // Model parameters.
  public final int numTopics;            // total topics including junk topics
  public final int numJunkTopics;        // number of junk topics (does not increase the numTopics count)
  public final int numTemplates;         // total number of templates, if using a nested template->topic->mention model
  public final int numJunkTemplates;     // number of junk templates (does not increase the numTemplates count)
  private final int numTopicsPerTemplate;
  private final double topicSmoothing = .1;
  private final double junkTopicSmoothing = 20.0;
  double wSmoothing = 3.0;
  double depSmoothing = 0.1;
  double verbSmoothing = 0.1;
  double featSmoothing = 1.0;
  private final double topicSmoothingTimesNumTopics;
  private double wSmoothingTimesNumW;
  private double depSmoothingTimesNumDeps;
  private double verbSmoothingTimesNumVerbs;
  private final double featSmoothingTimesNumFeats;

  double _lastLikelihood = -Double.MAX_VALUE;
  double _lastLikelihoodDelta = -Double.MAX_VALUE;
  double _bestLikelihood = -Double.MAX_VALUE;
  int _bestLikelihoodStep = 0;
  int currentIteration = -1; // keep track of what iteration the sampler is on
  EntityModelInstance _bestModelInstance;

  public GibbsSamplerEntities() {
    this(10, 0, 2, 0);
  }

  public GibbsSamplerEntities(int numTopics) {
  	this(numTopics, 0, 2, 0);
  }
  
  public GibbsSamplerEntities(int numTopics, int numJunkTopics, int numTemplates, int numJunkTemplates) {
    this.numTopics = numTopics;
    this.numTemplates = numTemplates;
    this.numTopicsPerTemplate = (numTemplates == 0 ? 0 : numTopics / numTemplates);
    this.numJunkTopics = (numJunkTemplates > 0 ? numTopicsPerTemplate*numJunkTemplates : numJunkTopics);
    this.numJunkTemplates = numJunkTemplates;
    this.topicSmoothingTimesNumTopics = (double)(this.numTopics-this.numJunkTopics) * topicSmoothing + ((double)this.numJunkTopics * junkTopicSmoothing);
    this.featSmoothingTimesNumFeats = (double)numFeats * featSmoothing;
    //    this.random = new Random(123876473); // give it a random number as the seed
    this.random = new Random(); // give it a random number as the seed

    System.out.printf("Sampler started with %d topics (%.1f smoothing, %d junk), %d templates (%d junk), %.1f summed topic smoothing\n", 
    		this.numTopics, this.topicSmoothing, this.numJunkTopics, this.numTemplates, this.numJunkTemplates, this.topicSmoothingTimesNumTopics);

    if( numTopics > 0 && numTemplates > 0 && numTopics % numTemplates != 0 ) {
      System.out.println("Nested model ERROR: number of topics " + numTopics + " is not divisible by number of templates " + numTemplates);
      System.exit(1);
    }
    
    _bestModelInstance = new EntityModelInstance();
  }

  private void printGlobals() {
    System.out.println("\tThetas per Doc: " + thetasInDoc);
    System.out.println("\tInclude Entity Features: " + includeEntityFeatures);
    System.out.println("\tInclude Verbs: " + includeVerbs);
    System.out.println("\tConstrain Subj/Obj Inclusion: " + constrainInverseDeps);
    System.out.printf("\tTopic Smoothing = %.2f\n", topicSmoothing);
    System.out.printf("\tWord Smoothing  = %.2f\n", wSmoothing);
    System.out.printf("\tDep Smoothing   = %.2f\n", depSmoothing);
    System.out.printf("\tVerb Smoothing   = %.2f\n", verbSmoothing);
    System.out.printf("\tFeat Smoothing  = %.2f\n", featSmoothing);
  }

  /**
   * Initializes the model with the given document's entities. An entity contains its entity
   * mentions which are simply words and typed dependencies. 
   * @param docEntities A list of entity objects, each entity contains a list of mentions.e
   */
  public void initializeModelFromData(List<String> docnames, List<List<TextEntity>> docEntities) {
    System.out.println("initializeModelFromData (" + docEntities.size() + " docs)");
    printGlobals();
    long startTime = System.currentTimeMillis();
    int numDocs = docEntities.size();
    this.docNames = docnames;
    TextEntity.TYPE[] featTypes = TextEntity.TYPE.values();

    // debug out
    int counts = 0;
    for( List<TextEntity> doc : docEntities ) counts += doc.size();
    System.out.println("initializeModelFromData with " + counts + " entities across all docs.");
    
    // init tCountsByTopic       
    wCountsBySlot   = ErasureUtils.<ClassicCounter<Integer>>mkTArray(ClassicCounter.class,numTopics);
    verbCountsBySlot   = ErasureUtils.<ClassicCounter<Integer>>mkTArray(ClassicCounter.class,numTopics);
    depCountsBySlot = ErasureUtils.<ClassicCounter<Integer>>mkTArray(ClassicCounter.class,numTopics);
    featCountsBySlot = ErasureUtils.<ClassicCounter<Integer>>mkTArray(ClassicCounter.class,numTopics);
    for (int i = 0; i < numTopics; i++) {
      wCountsBySlot[i]   = new ClassicCounter<Integer>();
      verbCountsBySlot[i]   = new ClassicCounter<Integer>();
      depCountsBySlot[i] = new ClassicCounter<Integer>();
      featCountsBySlot[i] = new ClassicCounter<Integer>();
    }

    // create index,  zs, topicCountsByDocument, topicSmoothing, wSmoothing, random
    wordIndex   = new HashIndex<String>();
    verbIndex   = new HashIndex<String>();
    depIndex    = new HashIndex<String>();
    words       = new int[numDocs][][];
    deps        = new int[numDocs][][];
    inverseDeps = new int[numDocs][][];
    verbs       = new int[numDocs][][];
    feats       = new int[numDocs][][];
    zs          = new int[numDocs][];
    topicCounts = new double[numTopics];
    topicCountsByDoc = new int[numDocs][];

    // Initialize zs uniformly at random. Increment all counts based on these.
    int doci = 0;
    for (List<TextEntity> doc : docEntities) {
      //      System.out.println("doc " + doc);
      int numEntities = doc.size();
      words[doci] = new int[numEntities][];
      deps[doci]  = new int[numEntities][];
      inverseDeps[doci]  = new int[numEntities][];
      if( includeVerbs )
        verbs[doci] = new int[numEntities][];
      if( includeEntityFeatures )
        feats[doci]  = new int[numEntities][];
      zs[doci]    = new int[numEntities];
      topicCountsByDoc[doci] = new int[numTopics];

      int entityi = 0;
      for( TextEntity entity : doc ) {
        //        System.out.println("\t" + entity.toFullString());
        words[doci][entityi] = new int[entity.numMentions()];
        deps[doci][entityi]  = new int[entity.numMentions()];
        inverseDeps[doci][entityi]  = new int[entity.numMentions()];
        if( includeVerbs )
          verbs[doci][entityi]  = new int[entity.numMentions()];
        if( includeEntityFeatures ) 
          feats[doci][entityi]  = new int[numFeats];
        //        System.out.println("adding words doc=" + doci + "\tentity=" + entityi);

        // Choose a topic for this entity.
        int topic = random.nextInt(numTopics);
        zs[doci][entityi] = topic;
        topicCounts[topic]++;
        topicCountsByDoc[doci][topic]++;

        // Set the features for this entity.
        if( includeEntityFeatures ) {
          for( int feati = 0; feati < numFeats; feati++ ) {
            boolean containsFeat = entity.types.contains(featTypes[feati]);
            feats[doci][entityi][feati] = (containsFeat ? 1 : 0);
            if( containsFeat ) 
              featCountsBySlot[topic].incrementCount(feati);
          }
        }

        // Count the mentions.
        //        int wi = 0;
        for( int wi = 0; wi < entity.numMentions(); wi++ ) {
          String token = entity.getCoreToken();
          wordIndex.add(token);
          int wIndex = wordIndex.indexOf(token);
          String verb = entity.deps.get(wi);
          verb = verb.substring(verb.indexOf("--")+2);
          verbIndex.add(verb);
          int vIndex = verbIndex.indexOf(verb);
          String dep = entity.deps.get(wi);
          String inverseDep = inverseDep(entity.deps.get(wi));
          if( dep == null ) {
            System.out.println("ERROR: null deps dep=" + dep + " on entity " + entity);
            System.exit(-1);
          }
          depIndex.add(dep);
          int dIndex = depIndex.indexOf(dep);

          // If no inverse dep relation, then set it to -1
          int dinverseIndex = -1;
          if( inverseDep != null ) {
            depIndex.add(inverseDep);
            dinverseIndex = depIndex.indexOf(inverseDep);
          }

          words[doci][entityi][wi] = wIndex;
          deps[doci][entityi][wi] = dIndex;
          inverseDeps[doci][entityi][wi] = dinverseIndex;
          if( includeVerbs ) verbs[doci][entityi][wi] = vIndex;

          wCountsBySlot[topic].incrementCount(wIndex);
          if( includeVerbs ) verbCountsBySlot[topic].incrementCount(vIndex);
          depCountsBySlot[topic].incrementCount(dIndex);
          numMentionsInAllDocs++;
          //          System.out.println("added word " + wIndex + " " + wordIndex.get(wIndex) + " to topic=" + topic + "\tentityi=" + entityi + "\twi=" + wi + "\t\t" + entity);
          //          wi++;
        } // entity mentions
        entityi++;
        numEntitiesInAllDocs++;
      } // entities
      
      // Check topic counts per doc.
//    	int sum = 0;
//      	for( int xx = 0; xx < numTopics; xx++ )
//      		sum += topicCountsByDoc[doci][xx];
//      	if( sum != docEntities.size() )
//      		System.out.println("sum=" + sum + "\tnumentities=" + docEntities.get(doci).size());
      	
      //      printDocument(doci);
      doci++;
    } // docs
    System.out.println("Finished init. wordIndex.size()=" + wordIndex.size() + " depIndex.size()=" + depIndex.size() + " verbIndex.size()=" + verbIndex.size());
    System.out.println("\tNum total entities = " + numEntitiesInAllDocs);

    wSmoothingTimesNumW = wSmoothing * (double)wordIndex.size();
    depSmoothingTimesNumDeps = depSmoothing * (double)depIndex.size();
    verbSmoothingTimesNumVerbs = verbSmoothing * (double)verbIndex.size();

    Util.reportElapsedTime(startTime);
    if( !checkVerbDistributions() ) System.exit(1);
    if( !checkTopicDistributions() ) System.exit(1);
  }

  /**
   * @return A probability distribution, not in log-space.
   */
  public double[] getTopicDistribution(int doc, int entityPosition) {
    int[] mentionTokens = words[doc][entityPosition];
    int[] mentionVerbs  = (includeVerbs ? verbs[doc][entityPosition] : null);
    int[] mentionDeps   = deps[doc][entityPosition];
    int[] mentionInverseDeps = inverseDeps[doc][entityPosition];
    int[] features      = (includeEntityFeatures ? feats[doc][entityPosition] : null);
    int numMentions = mentionTokens.length;
    
    double[] probs = new double[numTopics];
    for (int topic = 0; topic < numTopics; topic++) {

      double probOfTopic = (thetasInDoc ? probOfTopicGivenDoc(topic, doc) : probOfTopic(topic));
      // This assumes all mentions use the same core entity token.
      double probOfWGivenTopic = (wCountsBySlot[topic].getCount(mentionTokens[0]) + wSmoothing) / (wCountsBySlot[topic].totalCount() + wSmoothingTimesNumW);
      probs[topic] = Math.log(probOfTopic * probOfWGivenTopic);

      //      if( topic == 0 ) {
      //      	double sum = 0.0; int countSum = 0; int times = 0;
      //      	for( int ii = 0; ii < wordIndex.size(); ii++ ) {	
      //      		double count = wCountsBySlot[topic].getCount(ii);
      //      		double totalCount = (double)wCountsBySlot[topic].totalCount();
      //      		double probW = (count + wSmoothing) / (totalCount + wSmoothingTimesNumW);
      ////      		System.out.printf("  %.2f + %.2f / %.2f + %.2f\n", count, depSmoothing, totalCount, depSmoothingTimesNumDeps);
      //      		sum += probW;
      //      		countSum += count;
      //      		times++;
      //      		if( count > 0.0 || probW > 0.0 ) 
      //      			System.out.printf("%d.\td=%s\tcount=%d/%d\tprob(d|topic)=%.5f\twSmoothing=%.2f\twSmoothTimesNumW=%.2f\n", ii, wordIndex.get(ii), (int)count, (int)totalCount, probW, wSmoothing, wSmoothingTimesNumW);
      //      	}
      //      	System.out.println("Looped " + times + " times.");
      //      	System.out.println("*****countSum = " + countSum);
      //      	System.out.println("*****sum = " + sum);
      //      	System.exit(-1);
      //      }

      // Calculate the Top-Level features e.g., P(physobject | topic)
      //      if( includeEntityFeatures ) {
      //        for( int feat = 0; feat < numFeats; feat++ ) {
      //          if( features[feat] == 1) {
      //            double probFeatGivenTopic = (featCountsBySlot[topic].getCount(feat) + featSmoothing) / (featCountsBySlot[topic].totalCount() + featSmoothingTimesNumFeats);
      //            probs[topic] += Math.log(probFeatGivenTopic);
      //          }
      //        }
      //      }

      // This version appears similar in performance to the above one (no interpolation).
      if( includeEntityFeatures ) {
        for( int feat = 0; feat < numFeats; feat++ ) {
          double sumprobs = 0.0;
          int numOn = 0;
          if( features[feat] == 1) {
            sumprobs += (featCountsBySlot[topic].getCount(feat) + featSmoothing) / (featCountsBySlot[topic].totalCount() + featSmoothingTimesNumFeats);
            numOn++;
          }
          if( numOn > 0 ) // same as interpolating with uniform lambda weights on each probability
            probs[topic] += Math.log(sumprobs / numOn);
        }
      }

      // Multiplies in F probabilities, one for each "entity feature".
      //      if( includeEntityFeatures ) {
      //        for( int feat = 0; feat < numFeats; feat++ ) {
      //          double probFeatGivenTopic;
      //          double count = featCountsBySlot[topic].getCount(feat);
      //          if( features[feat] == 1)
      //            probFeatGivenTopic = (count + featSmoothing) / (topicCounts[topic] + featSmoothingTimesNumFeats);
      //          else
      //            probFeatGivenTopic = (topicCounts[topic] - count + featSmoothing) / (topicCounts[topic] + featSmoothingTimesNumFeats);
      //          probs[topic] += Math.log(1.0/7.0 * probFeatGivenTopic);
      //        }
      //      }

      // Loop over the mentions for this entity.
      for( int mention = 0; mention < numMentions; mention++ ) {
        //        String word = this.wordIndex.get(mentionTokens[mention]);
        //        String dep = this.depIndex.get(mentionDeps[mention]);

        //        double probOfDepGivenTopic = Math.log(depCountsBySlot[topic].getCount(mentionDeps[mention]) + depSmoothing) - Math.log(depCountsBySlot[topic].totalCount() + depSmoothingTimesNumDeps);

        // Divide is faster than logarithm (or so the internet seems to think)
        double probOfDepGivenTopic = (depCountsBySlot[topic].getCount(mentionDeps[mention]) + depSmoothing) / (depCountsBySlot[topic].totalCount() + depSmoothingTimesNumDeps);

        // Penalty if the nsubj is much higher than the dobj in this topic.
        if( constrainInverseDeps && inverseDeps != null && currentIteration > 50 ) {
          // Only nsubj and dobj have inverses, so things like prep_in are null (value -1 in the array).
          if( mentionInverseDeps[mention] >= 0 ) {
            double probOfInverseDepGivenTopic = (depCountsBySlot[topic].getCount(mentionInverseDeps[mention]) + depSmoothing) / (depCountsBySlot[topic].totalCount() + depSmoothingTimesNumDeps);
            if( probOfInverseDepGivenTopic > probOfDepGivenTopic+0.03 ) {
              //            System.out.printf("Higher topic %d, %s: %.4f\t%.4f\n", topic, depIndex.get(mentionDeps[mention]), probOfInverseDepGivenTopic, probOfDepGivenTopic);
              probOfDepGivenTopic = .0001;
            }
          }
        }

        probOfDepGivenTopic = Math.log(probOfDepGivenTopic);

        //        System.out.printf("  %s dep  count %.1f total dep count %.1f, depsmoothing=%.1f, deptotalsmoothing=%.1f\n", 
        //            dep, depCountsBySlot[topic].getCount(mentionDeps[mention]), depCountsBySlot[topic].totalCount(), depSmoothing, depSmoothingTimesNumDeps);
        //        System.out.printf("  %s word count %.1f total word count %.1f, wsmoothing=%.1f, wtotalsmoothing=%.1f\n", 
        //            word, wCountsBySlot[topic].getCount(mentionTokens[mention]), wCountsBySlot[topic].totalCount(), wSmoothing, wSmoothingTimesNumW);
        probs[topic] += probOfDepGivenTopic;


        if( includeVerbs ) {
          double probOfVerbGivenTopic = probOfVerbGivenTopic(mentionVerbs[mention], topic);
          probs[topic] += Math.log(probOfVerbGivenTopic);
//          System.out.printf("v=%s\tt=%d\tp=%.4f\n", verbIndex.get(mentionVerbs[mention]), topic, probOfVerbGivenTopic);
        }


        // DEBUGGING OUTPUT
        //        System.out.printf("P(slot=%d)= %.5f * P(%s|slot=%d)= %.5f(%.5f) * P(%s|slot=%d)= %.5f(%.5f) \t= %.5f(%.9f)\n", 
        //            topic, probOfTopic, word, topic, Math.exp(probOfWGivenTopic), probOfWGivenTopic, dep, topic, Math.exp(probOfDepGivenTopic), probOfDepGivenTopic, probs[topic], Math.exp(probs[topic]));
      }
    }

    ArrayMath.logNormalize(probs);
    ArrayMath.expInPlace(probs);

    //      ArrayMath.normalize(probs);
    //    System.out.println("Returning probs: " + Arrays.toString(probs));

    return probs;
  }

  public void unlabel(int doci, int entityi) {
    int[] mentionWords = words[doci][entityi];
    int[] mentionVerbs = (includeVerbs ? verbs[doci][entityi] : null);
    int[] mentionDeps  = deps[doci][entityi];
    int numMentions = mentionWords.length;
    int wordID = mentionWords[0];

    // Remove old counts of the current z.
    int oldZ = zs[doci][entityi];
    topicCounts[oldZ]--;
    topicCountsByDoc[doci][oldZ]--;

    wCountsBySlot[oldZ].decrementCount(wordID);
    if (SloppyMath.isCloseTo(wCountsBySlot[oldZ].getCount(wordID), 0.0))
    	wCountsBySlot[oldZ].remove(wordID);

    for (int mention = 0; mention < numMentions; mention++) {
    	int depID = mentionDeps[mention];

    	depCountsBySlot[oldZ].decrementCount(depID);
    	if (SloppyMath.isCloseTo(depCountsBySlot[oldZ].getCount(depID), 0.0))
    		depCountsBySlot[oldZ].remove(depID);

    	if( includeVerbs ) {
    		int verbID = mentionVerbs[mention];
    		verbCountsBySlot[oldZ].decrementCount(verbID);
    		if (SloppyMath.isCloseTo(verbCountsBySlot[oldZ].getCount(verbID), 0.0))
    			verbCountsBySlot[oldZ].remove(verbID);
    	}
    	//            System.out.println("subtracted " + wordID + " " + wordIndex.get(wordID) + " from z=" + oldZ);
    }

    if( includeEntityFeatures ) {
    	for( int feat = 0; feat < numFeats; feat++ ) {
    		if( feats[doci][entityi][feat] == 1 )
    			featCountsBySlot[oldZ].decrementCount(feat);
    	}
    }
  }

  public void relabel(int doci, int entityi, int newZ) {
  	int[] mentionWords = words[doci][entityi];
  	int[] mentionVerbs = (includeVerbs ? verbs[doci][entityi] : null);
  	int[] mentionDeps  = deps[doci][entityi];
  	int numMentions = mentionWords.length;
  	int wordID  = words[doci][entityi][0];

  	// Update counts with new sampled z value.
  	zs[doci][entityi] = newZ;
  	topicCounts[newZ]++;
  	topicCountsByDoc[doci][newZ]++;

  	wCountsBySlot[newZ].incrementCount(wordID);
  	for (int mention = 0; mention < numMentions; mention++) {
  		//          int wordID = mentionWords[mention];
  		int depID = mentionDeps[mention];
  		depCountsBySlot[newZ].incrementCount(depID);            

  		if( includeVerbs ) {
  			int verbID = mentionVerbs[mention];
  			verbCountsBySlot[newZ].incrementCount(verbID);            
  		}
  	}

  	if( includeEntityFeatures ) {
  		for( int feat = 0; feat < numFeats; feat++ ) {
  			if( feats[doci][entityi][feat] == 1 )
  				featCountsBySlot[newZ].incrementCount(feat);
  		}
  	}
  }

  /**
   * Assumes the documents have been given and loaded, ready to go.
   * Now run the gibbs sampling algorithm to determine values of slot indicator variables Z.
   * 
   * O( T * M ) - T num topics, M num mentions in entire corpus
   * 
   * @param numIterations The number of iterations to run.
   */
  public void runSampler(int numIterations) {
    // For each sample, we remove one entity's label and pretend it does not exist.
    // Then we sample the most likely label for it, so the probability calculation
    // should act like one entity is missing each time. I add this entity back in
    // at the end of this function.
    numEntitiesInAllDocs--; 

    for (int iter = 0; iter < numIterations; iter++) {
      System.err.println("Iteration: "+iter);
      currentIteration = iter;
      /*
      int kidnapTopic = -1;
      if( iter % 100 == 99 ) {
        kidnapTopic = findKidnapTopic();
        System.out.println("Kidnap topic: " + kidnapTopic);
      }
       */

      if( iter % 15 == 14 && stoppingCriterionLikelihoodMet(iter) )
        break;      
      if( iter % 15 == 14 ) {
      	numEntitiesInAllDocs++;
      	if( !checkVerbDistributions() || !checkTopicDistributions() )
      		System.exit(1);
      	numEntitiesInAllDocs--;
      }

      for (int doc = 0; doc < words.length; doc++) {
        
        for (int entity = 0; entity < words[doc].length; entity++) {
          //          System.out.println("doc=" + doc + "\tentity=" + entity);

        	unlabel(doc,entity);
        	/*
          int[] mentionWords = words[doc][entity];
          int[] mentionVerbs = (includeVerbs ? verbs[doc][entity] : null);
          int[] mentionDeps  = deps[doc][entity];
          int numMentions = mentionWords.length;

          // Remove old counts of the current z.
          int oldZ = zs[doc][entity];
          topicCounts[oldZ]--;
          topicCountsByDoc[doc][oldZ]--;

          int wordID = mentionWords[0];
          wCountsBySlot[oldZ].decrementCount(wordID);
          if (SloppyMath.isCloseTo(wCountsBySlot[oldZ].getCount(wordID), 0.0))
            wCountsBySlot[oldZ].remove(wordID);

          for (int mention = 0; mention < numMentions; mention++) {
            int depID = mentionDeps[mention];

            depCountsBySlot[oldZ].decrementCount(depID);
            if (SloppyMath.isCloseTo(depCountsBySlot[oldZ].getCount(depID), 0.0))
              depCountsBySlot[oldZ].remove(depID);

            if( includeVerbs ) {
              int verbID = mentionVerbs[mention];
              verbCountsBySlot[oldZ].decrementCount(verbID);
              if (SloppyMath.isCloseTo(verbCountsBySlot[oldZ].getCount(verbID), 0.0))
                verbCountsBySlot[oldZ].remove(verbID);
            }
            //            System.out.println("subtracted " + wordID + " " + wordIndex.get(wordID) + " from z=" + oldZ);
          }

          if( includeEntityFeatures ) {
            for( int feat = 0; feat < numFeats; feat++ ) {
              if( feats[doc][entity][feat] == 1 )
                featCountsBySlot[oldZ].decrementCount(feat);
            }
          }
*/
        	
          // Sample a new z.
          double[] probs = getTopicDistribution(doc, entity);
          int newZ = ArrayMath.sampleFromDistribution(probs);
          //          int newZ = random.nextInt(numTopics);

          // DEBUG
          //          for( int pp = 0; pp < probs.length; pp++ ) System.out.printf(" %.3f", probs[pp]);
          //          System.out.print(" oldz=" + oldZ + " newz=" + newZ);
          //          System.out.println();

          relabel(doc, entity, newZ);
          
          /*
          // Update counts with new sampled z value.
          zs[doc][entity] = newZ;
          topicCounts[newZ]++;
          topicCountsByDoc[doc][newZ]++;

          wCountsBySlot[newZ].incrementCount(wordID);
          for (int mention = 0; mention < numMentions; mention++) {
            //          int wordID = mentionWords[mention];
            int depID = mentionDeps[mention];
            depCountsBySlot[newZ].incrementCount(depID);            

            if( includeVerbs ) {
              int verbID = mentionVerbs[mention];
              verbCountsBySlot[newZ].incrementCount(verbID);            
            }
          }

          if( includeEntityFeatures ) {
            for( int feat = 0; feat < numFeats; feat++ ) {
              if( feats[doc][entity][feat] == 1 )
                featCountsBySlot[newZ].incrementCount(feat);
            }
          }
          */
        }
      }
    }

    numEntitiesInAllDocs++;
    //    System.out.println("Checking data structures result = " + checkDataStructures());
    
    // Load the best performer.
    System.out.println("Loading the best iteration from step " + _bestModelInstance.samplingStep + "...likelihood=" + _bestModelInstance.likelihood);
    loadBestModelInstance(_bestModelInstance);
  }

  //  private int[][] getTopicAssignments() {
  //    return ArrayUtils.copy(zs);
  //  }
  
  /**
   * Compute likelihood of the data to determine stopping point.
   */
  public boolean stoppingCriterionLikelihoodMet(int currentStep) {
    double like = computeDataLikelihood();
    double change = _lastLikelihood - like;
  
    if( Math.abs(change) > 999999 )
      System.out.printf("Likelihood = %.3f (big number)\n", like, change);
    else
      System.out.printf("Likelihood = %.3f (%.1f)\n", like, change);
    
    // Data likelihood is stable.
    if( currentStep > 1000 && Math.abs(change) < 20.0 && _lastLikelihoodDelta < 20.0 )
      return true;
  
    // We found a lower likelihood, and that lower likelihood was found a long time ago.
    int stepsAgo = currentStep - _bestLikelihoodStep;
    if( like < _bestLikelihood ) 
    	System.out.printf("Best %.1f found %d steps ago.\n", _bestLikelihood, stepsAgo);
    if( currentStep > 400 && _bestLikelihood > like && (stepsAgo > 1000 || (double)stepsAgo/(double)currentStep > 0.4) ) {
      System.out.println("Stopping criterion: better likelihood found a long time ago.");
      return true;
    }      
  
    // Found a better likelihood!
    if( like > _bestLikelihood ) {
    	System.out.println("** New best!");
      _bestLikelihood = like;
      _bestLikelihoodStep = currentStep;
      _bestModelInstance.storeAll(zs, topicCounts, topicCountsByDoc, wCountsBySlot, verbCountsBySlot, depCountsBySlot, featCountsBySlot);
      _bestModelInstance.likelihood = like;
      _bestModelInstance.samplingStep = currentStep;
    }
    
    _lastLikelihoodDelta = Math.abs(change);
    _lastLikelihood = like;    
    return false;
  }

  /**
   * Move the saved best sampling instance (by likelihood) into memory. Overwrite the current sampler's
   * z assignments and use the best model's counts/assignments.
   */
  public void loadBestModelInstance(EntityModelInstance best) {
  	System.out.println("Reloading best sampled instance...");
//    System.out.println("BEFORE LOADING BEST MODEL");
//    printWordDistributionsPerTopic();
    
    for( int xx = 0; xx < best.zs.length; xx++ )
      this.zs[xx] = Arrays.copyOf(best.zs[xx], best.zs[xx].length);
    
    for( int xx = 0; xx < best.topicCountsByDoc.length; xx++ )
      this.topicCountsByDoc[xx] = Arrays.copyOf(best.topicCountsByDoc[xx], best.topicCountsByDoc[xx].length);
    
    this.wCountsBySlot    = best.cloneCounter(best.wCountsBySlot);
    this.verbCountsBySlot = best.cloneCounter(best.verbCountsBySlot);
    this.depCountsBySlot  = best.cloneCounter(best.depCountsBySlot);
    this.featCountsBySlot = best.cloneCounter(best.featCountsBySlot);
    
//    System.out.println("\n\nAFTER LOADING BEST MODEL");
//    printWordDistributionsPerTopic();
  }

  private boolean isJunkTopic(int tt) {
  	if( tt >= (numTopics-numJunkTopics) ) return true;
  	else return false;
  }

  private boolean isInJunkTemplate(int topic) {
  	int template = topic / numTopicsPerTemplate;
  	if( template >= (numTemplates-numJunkTemplates) ) return true;
  	else return false;
  }

  public double probOfTopic(int topic) {
  	double topicSmoothing = this.topicSmoothing;
  	if( isJunkTopic(topic) ) topicSmoothing = this.junkTopicSmoothing;
  	return (topicCounts[topic] + topicSmoothing) / (numEntitiesInAllDocs + topicSmoothingTimesNumTopics);    
  }

  public double probOfTopicGivenDoc(int topic, int doc) {
  	if( numTemplates > 0 )
  		return probOfTemplateTopicGivenDoc(topic, doc);
  	
    int numentities = 0;
    for( int xx = 0; xx < numTopics; xx++ )
      numentities += topicCountsByDoc[doc][xx];
  	double topicSmoothing = this.topicSmoothing;
  	if( isJunkTopic(topic) ) topicSmoothing = this.junkTopicSmoothing;
//  	System.out.println("probOfTopicGivenDoc " + topic + " smooth=" + topicSmoothing);
  	
  	return (topicCountsByDoc[doc][topic] + topicSmoothing) / (numentities + topicSmoothingTimesNumTopics);    
  }

  /**
   * Calculate P(topic | doc) in the context of a template->topic nested model.
   * This is calculated as P(topic | template) * P(template | doc).
   * I make the simplifying assumption that P(topic | template) is uniform.
   * @param topic Index of the topic.
   * @param doc Index of the document. 
   * @return P(topic | document)
   */
  public double probOfTemplateTopicGivenDoc(int topic, int doc) {
    int numEntitiesInDoc = 0;
    for( int xx = 0; xx < numTopics; xx++ )
      numEntitiesInDoc += topicCountsByDoc[doc][xx];
  	double topicSmoothing = this.topicSmoothing;
  	if( isInJunkTemplate(topic) ) topicSmoothing = this.junkTopicSmoothing;
//  	System.out.println("probOfTopicGivenDoc " + topic + " smooth=" + topicSmoothing);

  	// Sum up the topics belonging to this template
  	Pair<Integer,Integer> startend = getSiblingTopics(topic);
  	int templateCount = 0;
  	for( int xx = startend.first(); xx < startend.second(); xx++ )
  		templateCount += topicCountsByDoc[doc][xx];
  	
  	// P(topic|doc) = Sum_template P(topic|doc)
  	// = Sum_template P(topic,template | doc)
  	// = Sum_template P(template|doc) * P(topic | template,doc)
  	// = Sum_template P(template|doc) * P(topic|template)
  	// = P(template|doc)*P(topic|template)   since P(topic|template)=0 for all other templates except its own
  	double probTopicGivenTemplate = 1.0 / numTopicsPerTemplate;
  	double probTemplateGivenDoc = (templateCount + (topicSmoothing*numTopicsPerTemplate)) / (numEntitiesInDoc + topicSmoothingTimesNumTopics); 

  	// P(topic|template) * P(template|doc)
  	return probTopicGivenTemplate * probTemplateGivenDoc;
  }
  
  public double probOfWGivenTopic(String w, int topic) {
    int tokenIndex = wordIndex.indexOf(w);
    return (wCountsBySlot[topic].getCount(tokenIndex) + wSmoothing) / (wCountsBySlot[topic].totalCount() + wSmoothingTimesNumW);
  }

  public double probOfDepGivenTopic(String dep, int topic) {
    int depi = depIndex.indexOf(dep);
    return (depCountsBySlot[topic].getCount(depi) + depSmoothing) / (depCountsBySlot[topic].totalCount() + depSmoothingTimesNumDeps);
  }

  public double probOfVerbGivenTopic(String verb, int topic) {
    return probOfVerbGivenTopic(verbIndex.indexOf(verb), topic);
  }
  public double probOfVerbGivenTopic(int verbID, int topic) {
    if( numTemplates > 0 )
      return probOfVerbGivenNestedTopic(verbID, topic);
    else
      return (verbCountsBySlot[topic].getCount(verbID) + verbSmoothing) / (verbCountsBySlot[topic].totalCount() + verbSmoothingTimesNumVerbs);
  }

  /**
   * Shared verb distribution: multiple slots under one template type
   * This computes the probability of a verb given a template. It sums the verb's
   * counts in all topics under the one template.
   */
  public double probOfVerbGivenNestedTopic(int verbID, int topic) {
    Pair<Integer,Integer> startend = getSiblingTopics(topic);
    int verbCounts = 0;
    int topicsTotalVerbCount = 0;
    for( int xx = startend.first(); xx < startend.second(); xx++ ) {
      verbCounts += verbCountsBySlot[xx].getCount(verbID);
      topicsTotalVerbCount += verbCountsBySlot[xx].totalCount();
    }
      
//    System.out.println("nested! " + ((verbCounts + verbSmoothing) / (topicsTotalVerbCount + verbSmoothingTimesNumVerbs)));
    return (verbCounts + verbSmoothing) / (topicsTotalVerbCount + verbSmoothingTimesNumVerbs);
  }

  /**
   * Returns a (start,end) pair representing the topic indices that are part of the template
   * that the given topic is part of. The end index is exclusive: [start,end)
   * @param topic The target topic.
   * @return A (start,end) pair of topics that are part of the same template that this topic is in.
   */
  private Pair<Integer,Integer> getSiblingTopics(int topic) {
    int mod = topic % numTopicsPerTemplate;
    int start = topic - mod;
    int end = topic + (numTopicsPerTemplate - mod);
    
    return new Pair<Integer,Integer>(start, end);
  }
  
  public double probOfFeatsGivenTopic(Set<TextEntity.TYPE> feats, int topic) { 
    TextEntity.TYPE[] featTypes = TextEntity.TYPE.values();

    double prob = 0.0;

    // Old way.
    //    for( int feati = 0; feati < featTypes.length; feati++ ) {
    //      if( feats.contains(featTypes[feati]) ) {
    //        double probFeatGivenTopic = (featCountsBySlot[topic].getCount(feati) + featSmoothing) / (featCountsBySlot[topic].totalCount() + featSmoothingTimesNumFeats);
    //        prob += Math.log(probFeatGivenTopic);
    //      }
    //    }

    for( int feat = 0; feat < featTypes.length; feat++ ) {
      int numOn = 0;
      double sumprobs = 0.0;
      if( feats.contains(featTypes[feat]) ) {
        sumprobs += (featCountsBySlot[topic].getCount(feat) + featSmoothing) / (featCountsBySlot[topic].totalCount() + featSmoothingTimesNumFeats);
        numOn++;
      }
      if( numOn > 0 ) // same as interpolating with uniform lambda weights on each probability
        prob = sumprobs / numOn;
    }    

    return prob;
  }

  /**
   * Analysis and Debugging
   * @param wordIndex The index of words to integer IDs.
   */
  public List<double[]> getWordDistributionsPerTopic(ClassicCounter<Integer>[] countsBySlot, double smoothing, double smoothingTimesNum, Index<String> wordIndex) {
    //    System.out.println("Calling getWordDistPerTopic...wordIndex size " + wordIndex.size());
    List<double[]> dists = new ArrayList<double[]>(numTopics);
    for( int topic = 0; topic < numTopics; topic++ ) {
      double[] dist = new double[wordIndex.size()];
      dists.add(dist);

      for( int ii = 0; ii < wordIndex.size(); ii++ ) {
        double probOfWGivenTopic = (countsBySlot[topic].getCount(ii) + smoothing) / (countsBySlot[topic].totalCount() + smoothingTimesNum);
        //        System.out.println("P(w=" + wordIndex.get(ii) + "|slot=" + topic + ") \t= " + probOfWGivenTopic);
        dist[ii] = probOfWGivenTopic;
      }
    }
    return dists;
  }

  public List<double[]> getFeatDistributionsPerTopic() {
    List<double[]> dists = new ArrayList<double[]>(numTopics);
    for( int topic = 0; topic < numTopics; topic++ ) {
      double[] dist = new double[numFeats];
      dists.add(dist);

      for( int ii = 0; ii < numFeats; ii++ ) {
        //        double probOfFeatGivenTopic = (featCountsBySlot[topic].getCount(ii) + featSmoothing) / 
        //        (featCountsBySlot[topic].totalCount() + featSmoothingTimesNumFeats);

        double probOfFeatGivenTopic = (featCountsBySlot[topic].getCount(ii) + featSmoothing) / (topicCounts[topic] + featSmoothingTimesNumFeats);;

        //        System.out.println("P(feat=" + ii + "|slot=" + topic + ") \t= " + probOfFeatGivenTopic);
        dist[ii] = probOfFeatGivenTopic;
      }
    }
    return dists;
  }


  public void printDocument(int docindex) {
    System.out.println("**Document " + docindex + "**");
    for( int entityi = 0; entityi < words[docindex].length; entityi++ ) {
      System.out.print("entity " + entityi);
      for( int mentioni = 0; mentioni < words[docindex][entityi].length; mentioni++ )
        System.out.println("\t" + wordIndex.get(words[docindex][entityi][mentioni]) + "," + depIndex.get(deps[docindex][entityi][mentioni]));
    }
    System.out.println();
  }

  /**
   * This takes the dependency relations (e.g., dobj--kidnap) and sums up the probabilities of
   * each base predicate (e.g., kidnap). It keeps the predicates with the highest sums.
   * @return A list of the top 5 predicates.
   */
  public List<String> getTopPredicatesBasedOnDeps() {
    Counter<String> depProbSums = new ClassicCounter<String>();
    List<double[]> depDistributions  = getWordDistributionsPerTopic(depCountsBySlot, depSmoothing, depSmoothingTimesNumDeps, depIndex);

    // Loop over each topic.
    for( int topicnum = 0; topicnum < numTopics; topicnum++ ) {
      double[] depDist = depDistributions.get(topicnum);

      // Build a map from dependency strings to their probabilities in this topic.
      Map<String,Double> depProbs = new HashMap<String,Double>(depDist.length*2);
      for( int ii = 0; ii < depDist.length; ii++ )
        depProbs.put(depIndex.get(ii), depDist[ii]);
      List<String> sortedDeps = Util.sortKeysByValues(depProbs,false);

      for( int ii = 0; ii < 5 && ii < sortedDeps.size(); ii++ ) {
        try {
          String dep = sortedDeps.get(ii);
          String predicate = dep.substring(dep.indexOf("--")+2);
          double prob = depProbs.get(dep);
          depProbSums.incrementCount(predicate, prob);
        } catch( Exception ex ) {
          System.out.println("Error with ii=" + ii + " and dep=" + sortedDeps.get(ii));
          System.out.println("sortedDeps size=" + sortedDeps.size() + " and depDist.length=" + depDist.length);
          ex.printStackTrace();
          System.exit(-1);
        }
      }
    }

    // Sort the used predicates.
    List<String> keys = Util.sortCounterKeys(depProbSums);
    //    for( String key : keys )
    //      System.out.println("predicate " + key + "\t=\t" + depProbSums.getCount(key));
    return keys;
  }

  /**
   * If the current model has a verb variable for each topic (e.g., distribution over verbs like 'kidnap'),
   * then this function returns the list of most probable verbs in the given topic.
   * @param topicnum The topic of which we want verbs.
   * @param n The number of verbs to return per topic AT A MINUMUM. It will always return at least this many (if there are that many in the topic).
   * @param minprob Any verbs above this probability will be returned.
   * @return A sorted list of the top n verbs and any other verbs over minprob in probability.
   */
  public List<String> getTopVerbsInTopic(int topicnum, int n, double minprob) {
    List<double[]> verbDistributions = (verbCountsBySlot == null ? null : getWordDistributionsPerTopic(verbCountsBySlot, verbSmoothing, verbSmoothingTimesNumVerbs, verbIndex));
    double[] verbDist = (verbDistributions == null ? null : verbDistributions.get(topicnum));

    // Build a map from verbs to probs in topic.
    List<String> sortedVerbs = null;
    Map<String,Double> verbProbs = null;
    if( verbDist != null ) {
      verbProbs = new HashMap<String,Double>(verbDist.length*2);
      for( int ii = 0; ii < verbDist.length; ii++ )
        verbProbs.put(verbIndex.get(ii), verbDist[ii]);
      // Sort the words by probability.
      sortedVerbs = Util.sortKeysByValues(verbProbs,false);
    }

    // Keep the top n and any others over a minimum probability.
    List<String> topVerbs = new ArrayList<String>();
    for( int ii = 0; ii < n && ii < sortedVerbs.size(); ii++ )
      topVerbs.add(sortedVerbs.get(ii));
    for( int ii = n; ii < sortedVerbs.size(); ii++ )
      if( verbProbs.get(sortedVerbs.get(ii)) >= minprob )
        topVerbs.add(sortedVerbs.get(ii));
      else break;

    return topVerbs;
  }

  /**
   * Given a topic num, figure out its template and compute the top verb list for the overall template.
   * @param topicnum Target topic you're interested in.
   * @param n The number of verbs to return sorted.
   * @param minprob Don't return any verbs with less probability mass than this.
   * @return List of sorted top verbs.
   */
  public List<String> getTopVerbsInTemplate(int topicnum, int n, double minprob) {
  	// Print the verb distribution for this template.
    Map<String,Double> verbTemplateProbs = new HashMap<String,Double>();
  	for( int xx = 0; xx < verbIndex.size(); xx++ ) {
  		double prob = probOfVerbGivenNestedTopic(xx, topicnum);
  		verbTemplateProbs.put(verbIndex.get(xx), prob);
  	}
  	List<String> sortedVerbs = Util.sortKeysByValues(verbTemplateProbs,false);
  	
    // Keep the top n and any others over a minimum probability.
    List<String> topVerbs = new ArrayList<String>();
    for( int ii = 0; ii < n && ii < sortedVerbs.size(); ii++ )
    	topVerbs.add(sortedVerbs.get(ii));
    for( int ii = n; ii < sortedVerbs.size(); ii++ )
    	if( verbTemplateProbs.get(sortedVerbs.get(ii)) >= minprob )
        topVerbs.add(sortedVerbs.get(ii));
      else break;

    return topVerbs;
  }
  
  // FOR DEBUGGING
  private void printDepIndex() {
    System.out.print("DEP INDEX DUMP: ");
    for( int xx = 0; xx < depIndex.size(); xx++ )
      System.out.print(xx + "-" + depIndex.get(xx) + " ");
    System.out.println();
  }

  public int[] countTopicOccurrences() {
  	int[] counts = new int[numTopics];
  	for( int doci = 0; doci < zs.length; doci++ )
  		for( int entityi = 0; entityi < zs[doci].length; entityi++ )
  			counts[zs[doci][entityi]]++;
  	return counts;
  }
  
  public int[][] getZs() {
  	return zs;
  }
  
  /**
   * This sampler has sampled n documents, and you might want to retrieve the labels from a 
   * particular document by name. We painfully O(n) search the list of names for the index. 
   * @param docname Name of the original document name that was used during sampling.
   * @return An index that is the position of the document in the original list of documents given to the sampler.
   */
  public int docnameToIndex(String docname) {
    int doci = 0;
    for( String name : docNames ) {
      if( name.equalsIgnoreCase(docname) )
        return doci;
      doci++;
    }
    return -1;
  }
  
  /**
   * Analysis and Debugging
   * This prints out each topic and a sorted list of the most probable tokens.
   * @param wordIndex The index of words to integer IDs.
   */
  public void printWordDistributionsPerTopic() {
    List<double[]> wordDistributions = getWordDistributionsPerTopic(wCountsBySlot, wSmoothing, wSmoothingTimesNumW, wordIndex);
    List<double[]> depDistributions  = getWordDistributionsPerTopic(depCountsBySlot, depSmoothing, depSmoothingTimesNumDeps, depIndex);
    List<double[]> featDistributions = getFeatDistributionsPerTopic();
    List<double[]> verbDistributions = (verbCountsBySlot == null ? null : getWordDistributionsPerTopic(verbCountsBySlot, verbSmoothing, verbSmoothingTimesNumVerbs, verbIndex));
    int[] topicCounts = countTopicOccurrences();

    for( int topicnum = 0; topicnum < numTopics; topicnum++ ) {
      double[] wordDist = wordDistributions.get(topicnum);
      double[] depDist = depDistributions.get(topicnum);
      double[] verbDist = (verbDistributions == null ? null : verbDistributions.get(topicnum));

      //printDepIndex();
      System.out.println("topic " + topicnum + " depDist length " + depDist.length + " verbDist length " + (verbDist == null ? null : verbDist.length));

      // Build a map from token strings to their probabilities in this topic.
      Map<String,Double> wordProbs = new HashMap<String,Double>(wordDist.length*2);
      for( int ii = 0; ii < wordDist.length; ii++ ) {
        wordProbs.put(wordIndex.get(ii), wordDist[ii]);
//        System.out.println(wordIndex.get(ii) + " is " + wordDist[ii]);        
      }
      
      // Sort the words by probability.
      List<String> sortedTokens = Util.sortKeysByValues(wordProbs,false);
//      System.out.println("Sorted tokens: ");
//      for( String token : sortedTokens )
//      	System.out.println("  - " + token + " " + wordProbs.get(token));

      // Build a map from dependency strings to their probabilities in this topic.
      Map<String,Double> depProbs = new HashMap<String,Double>(depDist.length*2);
      for( int ii = 0; ii < depDist.length; ii++ )
        depProbs.put(depIndex.get(ii), depDist[ii]);
      // Sort the words by probability.
      List<String> sortedDeps = Util.sortKeysByValues(depProbs,false);

      // Build a map from verbs to probs in topic.
      List<String> sortedVerbs = null;
      Map<String,Double> verbProbs = null;
      if( verbDist != null ) {
        verbProbs = new HashMap<String,Double>(verbDist.length*2);
        for( int ii = 0; ii < verbDist.length; ii++ )
          verbProbs.put(verbIndex.get(ii), verbDist[ii]);
        // Sort the words by probability.
        sortedVerbs = Util.sortKeysByValues(verbProbs,false);
      }

      // Print template header.
      if( numTemplates > 0 && topicnum == getSiblingTopics(topicnum).first() ) {
      	Pair<Integer,Integer> startend = getSiblingTopics(topicnum);
      	int seen = 0;
      	for( int ii = startend.first(); ii < startend.second(); ii++ )
      		seen += topicCounts[ii];
      	System.out.println("*** Template " + (topicnum/numTopicsPerTemplate) + " seen=" + seen + " ***");

      	// Print the verb distribution for this template.
      	List<String> sortedTemplateVerbs = getTopVerbsInTemplate(topicnum, 40, 0.0);
        for( int ii = 0; ii < sortedTemplateVerbs.size() && ii < 20; ii++ ) {
            String token = sortedTemplateVerbs.get(ii);
            String token20 = (ii+20 < sortedTemplateVerbs.size() ? sortedTemplateVerbs.get(ii+20) : "");
            double tprob = probOfVerbGivenNestedTopic(verbIndex.indexOf(token), topicnum);
            double t20prob = probOfVerbGivenNestedTopic(verbIndex.indexOf(token20), topicnum);

            if( token.length() < 8 )
            	System.out.printf("%s\t\t%.5f\t\t", token, tprob);
            else
            	System.out.printf("%s\t%.5f\t\t", token, tprob);
            if( token20.length() < 8 )
            	System.out.printf("%s\t\t%.5f\n", token20, t20prob);
            else
            	System.out.printf("%s\t%.5f\n", token20, t20prob);
        }
      }
      
      // Print topic header info.
      double probOfTopic = (thetasInDoc ? 0.0 : probOfTopic(topicnum));
      System.out.printf("*** Topic %d : p(topic=%d) = %.4f : seen=%d ***\n", topicnum, topicnum, probOfTopic, topicCounts[topicnum]);
      System.out.print("  - feats [ ");
      for( int feat = 0; feat < numFeats; feat++ )
        System.out.print(TextEntity.TYPE.values()[feat] + " ");
      System.out.print(" ]\n  - feats [ ");
      for( int feat = 0; feat < numFeats; feat++ )
        System.out.printf(" %.5f", featDistributions.get(topicnum)[feat]);
      System.out.println(" ]");

      // Print the top words and deps in this topic.
      for( int ii = 0; ii < sortedTokens.size() && ii < 25; ii++ ) {
        String token = sortedTokens.get(ii);
        String dep   = (ii < sortedDeps.size() ? sortedDeps.get(ii) : null);
        String verb  = ((sortedVerbs != null && ii < sortedVerbs.size()) ? sortedVerbs.get(ii) : null);

        if( token.length() < 8 )
          System.out.printf("%s\t\t%.5f\t\t", token, wordProbs.get(token));
        else 
          System.out.printf("%s\t%.5f\t\t", token, wordProbs.get(token));

        if( dep != null ) {
          if( dep.length() > 15 )
            System.out.printf("%s\t%.5f\t\t", dep, depProbs.get(dep));
          else if( dep.length() >= 8 )
            System.out.printf("%s\t\t%.5f\t\t", dep, depProbs.get(dep));
          else
            System.out.printf("%s\t\t\t%.5f\t\t", dep, depProbs.get(dep));
        }

        if( verb != null ) {
          if( verb.length() > 15 )
            System.out.printf("%s\t%.5f", verb, verbProbs.get(verb));
          else if( verb.length() >= 8 )
            System.out.printf("%s\t\t%.5f", verb, verbProbs.get(verb));
          else
            System.out.printf("%s\t\t\t%.5f", verb, verbProbs.get(verb));
        }
        System.out.println();
      }
    }
  }

  private String inverseDep(String dep) {
    if( dep.startsWith("nsubj") )
      return "dobj--" + dep.substring(7);
    if( dep.startsWith("dobj") )
      return "nsubj--" + dep.substring(6);
    return null;
  }

  private boolean checkTopicDistributions() {
  	System.out.println("Checking topic distributions");
  	// Thetas per doc or nested template model.
  	if( numTemplates > 0 || thetasInDoc ) {
  		for( int doci = 0; doci < zs.length; doci++ ) {
  			double sum = 0.0;
  			for( int xx = 0; xx < numTopics; xx++ )
  				sum += probOfTopicGivenDoc(xx, doci);
  			if( Math.abs(sum - 1.0) > .00001 ) {
  				System.out.println("ERROR: Topic distribution for doc " + doci + " sums to " + sum);
  				return false;
  			}
  		}
  	}
  	// Global thetas.
  	else {
  		double sum = 0.0;
  		for( int xx = 0; xx < numTopics; xx++ )
  			sum += probOfTopic(xx);
  		if( Math.abs(sum - 1.0) > .00001 ) {
  			System.out.println("ERROR: Topic distribution sums to " + sum);
  			System.out.println(Arrays.toString(topicCounts));
  			System.out.println("Number of entities: " + numEntitiesInAllDocs);
  			return false;
  		}
  	}
  	return true;
  }
  
  private boolean checkVerbDistributions() {
  	System.out.println("Checking verb distributions per topic for " + verbIndex.size() + " verbs.");
    boolean allgood = true;
    for( int topic = 0; topic < numTopics; topic++ ) {
      double sum = 0.0;
      for( String verb : verbIndex ) {
        double prob = probOfVerbGivenTopic(verbIndex.indexOf(verb), topic);
        sum += prob;
      }
      if( Math.abs(sum - 1.0) > .00001 ) {
        System.out.println("ERROR: Topic " + topic + " sums to " + sum);
        allgood = false;
      }
    }
    return allgood;
  }
  
  /**
   * THIS NEEDS TO BE WRITTEN FOR ENTITIES
   * Check that if we run over the words/deps of all the documents right now (presumably after running
   * some iterations), counting how many words/deps appear will match up with the global counts.
   */
  private boolean checkDataStructures() {
    double[] countedTopicCounts = new double[numTopics];
    ClassicCounter<Integer>[] countedWCountsBySlot = ErasureUtils.<ClassicCounter<Integer>>mkTArray(ClassicCounter.class,numTopics);
    ClassicCounter<Integer>[] countedDepCountsBySlot = ErasureUtils.<ClassicCounter<Integer>>mkTArray(ClassicCounter.class,numTopics);
    for (int i = 0; i < countedWCountsBySlot.length; i++) {
      countedWCountsBySlot[i] = new ClassicCounter<Integer>();
      countedDepCountsBySlot[i] = new ClassicCounter<Integer>();
    }
    for (int docNum = 0; docNum < words.length; docNum++) {
      for (int entityNum = 0; entityNum < words[docNum].length; entityNum++) {
        for (int wordNum = 0; wordNum < words[docNum].length; wordNum++) {
          int topic = zs[docNum][entityNum];
          int word = words[docNum][entityNum][wordNum];
          int dep = deps[docNum][entityNum][wordNum];
          countedTopicCounts[topic]++;
          countedWCountsBySlot[topic].incrementCount(word);
          countedDepCountsBySlot[topic].incrementCount(dep);
        }
      }
    }

    if( !Arrays.equals(countedTopicCounts, topicCounts) ) {
      System.out.println("Topic check failed: " + Arrays.toString(countedTopicCounts) + " actual " + Arrays.toString(topicCounts));
      return false;
    }

    if( !Arrays.equals(countedWCountsBySlot, wCountsBySlot) ) {
      System.out.println("WCounts check failed: " + Arrays.toString(countedWCountsBySlot) + " actual " + Arrays.toString(wCountsBySlot));
      return false;
    }

    if( !Arrays.equals(countedDepCountsBySlot, depCountsBySlot) ) {
      System.out.println("DepCounts check failed: " + Arrays.toString(countedDepCountsBySlot) + " actual " + Arrays.toString(depCountsBySlot));
      return false;
    }
    return true;
  }  

  public void toFile(String filename) {
    try {
      FileOutputStream f = new FileOutputStream(filename);
      ObjectOutputStream s = new ObjectOutputStream(f);
      s.writeObject(this);
      s.flush();
      f.close();
    } catch( Exception ex ) { ex.printStackTrace(); }
  }

  public static GibbsSamplerEntities fromFile(String filename) {
    try {
      FileInputStream f = new FileInputStream(filename);
      ObjectInputStream s = new ObjectInputStream(f);
      GibbsSamplerEntities sampler = (GibbsSamplerEntities)s.readObject();
      f.close();
      System.out.println("fromFile: numtopics    = " + sampler.numTopics);
      if( sampler.numTemplates > 0 ) System.out.println("fromFile: numtemplates = " + sampler.numTemplates);
      return sampler;
    } catch( Exception ex ) { ex.printStackTrace(); }
    return null;
  }

  /**
   * Assumes the documents have been given and loaded, ready to go.
   * Now run the gibbs sampling algorithm to determine values of slot indicator variables Z.
   * 
   * However, only so many entities can be mapped to each topic, determined by the global maxEntitiesPerTopic.
   * This function randomly chooses entities in a random order, and assigns topics in order until each
   * topic is full.
   * 
   * This ultimately does not work. The one topic that serves as the default just ends up being a language model
   * for the corpus. The other topics only get 2 mentions each, so the majority of mentions are thrown into the
   * same default topic, thus making a general LM. The specific topics never get to anything interesting because
   * the mentions that should be in the topic are not always randomly selected and instead have to be put in the
   * default topic.
   * 
   * @param numIterations The number of iterations to run.
   */
  public void runSamplerConstrained(int numIterations) {
    for (int iter = 0; iter < numIterations; iter++) {
      System.err.println("Iteration: "+iter);

      for (int doc = 0; doc < words.length; doc++) {
        int numEntities = words[doc].length;
        ArrayList<Integer> ordered = createOrderedList(numEntities);
        int[] docTopicCounts = new int[numTopics];

        for (int entity = 0; entity < numEntities; entity++) {
          // Choose a random entity, don't traverse in order.
          int rand = random.nextInt(numEntities-entity);
          int targetEntity = ordered.get(rand);
          ordered.remove(rand);
          //            System.out.println("doc=" + doc + "\tentity=" + targetEntity);

          int[] mentionWords = words[doc][targetEntity];
          int[] mentionDeps  = deps[doc][targetEntity];
          int numMentions = mentionWords.length;

          // Remove old counts of the current z.
          int oldZ = zs[doc][targetEntity];
          topicCounts[oldZ]--;

          for (int mention = 0; mention < numMentions; mention++) {
            int wordID = mentionWords[mention];
            int depID = mentionDeps[mention];

            wCountsBySlot[oldZ].decrementCount(wordID);
            if (SloppyMath.isCloseTo(wCountsBySlot[oldZ].getCount(wordID), 0.0))
              wCountsBySlot[oldZ].remove(wordID);

            depCountsBySlot[oldZ].decrementCount(depID);
            if (SloppyMath.isCloseTo(depCountsBySlot[oldZ].getCount(depID), 0.0))
              depCountsBySlot[oldZ].remove(depID);

            //              System.out.println("subtracted " + wordID + " " + wordIndex.get(wordID) + " from z=" + oldZ);
          }

          if( includeEntityFeatures ) {
            for( int feat = 0; feat < numFeats; feat++ ) {
              if( feats[doc][targetEntity][feat] == 1 )
                featCountsBySlot[oldZ].decrementCount(feat);
            }
          }

          // Sample a new z.
          double[] probs = getTopicDistribution(doc, targetEntity);
          int newZ = numTopics-1;
          if( entity < numTopics*maxEntitiesPerTopic )
            newZ = sampleFromDistribution(probs, docTopicCounts, maxEntitiesPerTopic);
          docTopicCounts[newZ]++;
          //          int newZ = random.nextInt(numTopics);

          // DEBUG
          //            for( int pp = 0; pp < probs.length; pp++ ) System.out.printf(" %.3f", probs[pp]);
          //            System.out.print(" oldz=" + oldZ + " newz=" + newZ);
          //            System.out.println();

          // Update counts with new sampled z value.
          zs[doc][targetEntity] = newZ;
          topicCounts[newZ]++;

          for (int mention = 0; mention < numMentions; mention++) {
            int wordID = mentionWords[mention];
            int depID = mentionDeps[mention];
            wCountsBySlot[newZ].incrementCount(wordID);
            depCountsBySlot[newZ].incrementCount(depID);            
            //              System.out.println("added " + wordID + " " + wordIndex.get(wordID) + " dep " + depIndex.get(depID) + " to " + newZ);
          }

          if( includeEntityFeatures ) {
            for( int feat = 0; feat < numFeats; feat++ ) {
              if( feats[doc][targetEntity][feat] == 1 )
                featCountsBySlot[newZ].incrementCount(feat);
            }
          }
        }
      }
      printWordDistributionsPerTopic();
    }

    //    System.out.println("Checking data structures result = " + checkDataStructures());
  }


  private ArrayList<Integer> createOrderedList(int n) {
    ArrayList<Integer> list = new ArrayList<Integer>(n);
    for( int ii = 0; ii < n; ii++ )
      list.add(ii);
    return list;
  }

  /**
   * Samples from the distribution over values 0 through d.length given by d.
   * Assumes that the distribution sums to 1.0.
   *
   * @param dist the distribution to sample from
   * @return a value from 0 to d.length
   */
  public int sampleFromDistribution(final double[] dist, final int[] topicCounts, final int maxPerTopic) {
    //      System.out.println("topicCounts = " + Arrays.toString(topicCounts));
    // sample from the uniform [0,1]
    double r = random.nextDouble();
    // now compare its value to cumulative values to find what interval it falls in
    double total = 0;
    for (int i = 0; i < dist.length - 1; i++) {
      if (Double.isNaN(dist[i])) {
        throw new RuntimeException("Can't sample from NaN");
      }
      total += dist[i];
      if (r < total) {
        // Find the next i that isn't over counted in topicCounts.
        while( topicCounts[i] >= maxPerTopic ) {
          i++;
          if( i >= dist.length ) i = 0;
        }
        return i;
      }
    }
    return dist.length - 1; // in case the "double-math" didn't total to exactly 1.0
  }

  /**
   * Assumes the entities are labeled with topics/slots.
   */
  public double computeDataLikelihood() {
    double likelihood = 0.0;
    for( int doc = 0; doc < words.length; doc++ ) {
      for( int entity = 0; entity < words[doc].length; entity++ ) {
        // The current entity's label.
        int currentZ = zs[doc][entity];
        // This is inefficient ... we should just compute probability for this Z, not all Z's.
        double[] probs = getTopicDistribution(doc, entity);
        double prob = probs[currentZ];
        likelihood += Math.log(prob);
      }
    }
    return likelihood;
  }

}
