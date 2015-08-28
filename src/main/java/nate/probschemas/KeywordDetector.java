package nate.probschemas;

import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import edu.stanford.nlp.stats.ClassicCounter;
import edu.stanford.nlp.stats.Counter;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.trees.TypedDependency;

import nate.IDFMap;
import nate.ProcessedData;
import nate.util.Directory;
import nate.util.TreeOperator;
import nate.util.WordNet;
import nate.util.Ling;

/**
 * Code that reads a document(s) and gives you the top most important event words.
 * This should be used for IR to find similar documents based on these keywords.
 * 
 * This code is used by --KeywordTokenizer-- to process all of Gigaword and provide a short list
 * of keywords to be used by the Lucene index for search.
 * 
 * getKeywords() is the main function.
 *  - this was used to convert Gigaword docs
 *  - use this to reduce current docs, and then call IRDocuments with those words
 * 
 */
public class KeywordDetector {
  public WordNet wordnet = null;
  public IDFMap idf;

  public KeywordDetector() {
    init();
  }

  public KeywordDetector(String[] args) {
    init();
    runit(args);
  }

  private void init() {
  	wordnet = new WordNet(WordNet.findWordnetPath());    
    idf = new IDFMap(IDFMap.findIDFPath());
  }
  
  public void runit(String[] args) {
    load(args[args.length-1]);
  }
  
  /**
   * Just a testing function.
   */
  public void load(String dataDir) {
    String parsesFile = dataDir + File.separator + Directory.nearestFile("parse", dataDir);
    String depsFile   = dataDir + File.separator + Directory.nearestFile("deps", dataDir);
    String eventsFile = dataDir + File.separator + Directory.nearestFile("events", dataDir);
    String nerFile    = dataDir + File.separator + Directory.nearestFile("ner", dataDir);

    // Read the data files from disk.
    ProcessedData data = new ProcessedData(parsesFile, depsFile, eventsFile, nerFile);
    data.nextStory();
        
    // Count all of the verbs.
    while( data.getParseStrings() != null ) {
      List<Tree> trees = TreeOperator.stringsToTrees(data.getParseStrings());
      Counter<String> verbs = getKeywordCounts(trees, data.getDependencies());
      
      System.out.println("DOC " + data.currentStory());
      for( String key : verbs.keySet() )
        if( idf.get("v-" + key) > 1.5 )
          System.out.println("\t" + key + "\t" + verbs.getCount(key) + "\t" + idf.get("v-" + key));
        else 
          System.out.println("\t(skip) " + key + "\t" + verbs.getCount(key) + "\t" + idf.get("v-" + key));
      verbs.clear();
        
      data.nextStory();
    }
  }
  
  public List<String> getKeywords(List<Tree> trees, List<List<TypedDependency>> deps) {
    List<String> verbs = new ArrayList<String>();
    int sid = 0;

    if( trees.size() != deps.size() ) {
      System.out.println("ERROR: " + trees.size() + " trees but " + deps.size() + " deps in KeywordDetector.getKeywords()");
      return null;
    }
    
    for( Tree tree : trees ) {
      // Look for particles.
      Map<Integer,String> particles = Ling.particlesInSentence(deps.get(sid++));
      
      for( int ii = 1; ii <= TreeOperator.getTreeLength(tree); ii++ ) {
        String token = TreeOperator.indexToToken(tree, ii);
        String tag = TreeOperator.indexToPOSTag(tree, ii);
        // Only count verbs.
        if( tag.startsWith("VB") ) {
          String lemma = wordnet.lemmatizeTaggedWord(token.toLowerCase(), tag);
          // Skip common verbs.
          if( !DataSimplifier.isReportingVerbLemma(lemma) && !DataSimplifier.isCommonVerbLemma(lemma) && lemma.charAt(0) != '\'' ) {
            // Skip verbs that are super rare (these are typically mistagged non-verbs).
            if( idf.getDocCount("v-" + lemma) > 400 ) {
              // Append the particle if one exists.
              if( particles.containsKey(ii) )
                lemma += "_" + particles.get(ii);
              // Count this lemma.
              verbs.add(lemma);
            }
          }
        }
      }
    }
    return verbs;
  }
  
  /**
   * Just for testing counts right now...
   */
  private Counter<String> getKeywordCounts(List<Tree> trees, List<List<TypedDependency>> deps) {
    Counter<String> verbs = new ClassicCounter<String>();
    int sid = 0;
    
    for( Tree tree : trees ) {
      // Look for particles.
      Map<Integer,String> particles = Ling.particlesInSentence(deps.get(sid++));
      
      for( int ii = 1; ii <= TreeOperator.getTreeLength(tree); ii++ ) {
        String token = TreeOperator.indexToToken(tree, ii);
        String tag = TreeOperator.indexToPOSTag(tree, ii);
        // Only count verbs.
        if( tag.startsWith("VB") ) {
          String lemma = wordnet.lemmatizeTaggedWord(token.toLowerCase(), tag);
          if( !DataSimplifier.isReportingVerbLemma(lemma) && !DataSimplifier.isCommonVerbLemma(lemma) ) {
            // Append the particle if one exists.
            if( particles.containsKey(ii) )
              lemma += "_" + particles.get(ii);
            // Count this lemma.
            verbs.incrementCount(lemma);
          }
        }
      }
    }
    return verbs;
  }
  
  /**
   * @param args
   */
  public static void main(String[] args) {
    KeywordDetector kd = new KeywordDetector(args);
  }

}
