package nate.probschemas;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.io.RandomAccessFile;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import edu.stanford.nlp.stats.ClassicCounter;
import edu.stanford.nlp.stats.Counter;
import edu.stanford.nlp.trees.LabeledScoredTreeFactory;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.trees.TreeFactory;
import edu.stanford.nlp.trees.TypedDependency;

import nate.util.Directory;
import nate.util.Ling;
import nate.EntityMention;
import nate.IDFMap;
import nate.NERSpan;
import nate.util.Pair;
import nate.ProcessedData;
import nate.util.Locks;
import nate.util.TreeOperator;
import nate.util.Util;
import nate.util.WordNet;

/**
 * This class takes parse trees and dependency graphs, and strips them down into
 * just a list of Entities. The entire purpose is to take a document and produce
 * the list of main entities in it, annotated with their mentions and syntactic
 * contexts.
 *
 * Main functions: getEntityList(), getCachedEntityList()
 * 
 */
public class DataSimplifier {
  IDFMap generalIDF;
  TreeFactory _tf;
  WordNet _wordnet = null;
  boolean _changeTokensToNELabel = false; // if true, then words that are labeled by NER are changed to that NE label.
  private final String _cacheDir = "cache";
  private final String _cacheGigaDir = "cachegiga";
  private double _minDepCounts = 10; // number of times a dep must be seen
  private int _minDocCounts = 10;    // number of docs a verb must occur in
  public boolean debug = false;

  public DataSimplifier(int minDepCount, int minDocCount) {
    this();
    _minDepCounts = minDepCount;
    _minDocCounts = minDocCount;
    System.out.println("DataSimplifier minDepCounts = " + minDepCount);
    System.out.println("DataSimplifier minDocCounts = " + minDocCount);
  }
  
  public DataSimplifier() {
    _tf = new LabeledScoredTreeFactory();
    System.out.println("Loading Wordnet from: " + WordNet.findWordnetPath());
    _wordnet = new WordNet(WordNet.findWordnetPath());
    System.out.println("Loading IDF from: " + IDFMap.findIDFPath());
    if( IDFMap.findIDFPath() == null )
      System.out.println("ERROR: no path to the IDF file found in IDFMap.java findIDFPath()");
    generalIDF = new IDFMap(IDFMap.findIDFPath());
  }

  /**
   * Process all of Gigaword ahead of time and save to disk.
   */
  public void simplifyGigaword(String parseDir, String depDir, String entityDir, String nerDir) {
    for( String file : Directory.getFilesSorted(parseDir) ) {
      if( file.startsWith("apw_eng") || file.startsWith("nyt_eng") ) {
        String corefile = file.substring(0, 14);
        if( Locks.getLock("simplify-" + corefile) ) {
          System.out.println("Now " + corefile);
          String pPath = parseDir + File.separator + file;
          String dPath = depDir + File.separator + Directory.nearestFile(corefile, depDir);
          String ePath = entityDir + File.separator + Directory.nearestFile(corefile, entityDir);
          String nPath = nerDir + File.separator + Directory.nearestFile(corefile, nerDir);
          ProcessedData data = new ProcessedData(pPath, dPath, ePath, nPath);
          data.nextStory();

          List<String> docsNames = new ArrayList<String>();
          List<List<TextEntity>> docsEntities = getEntityList(data, docsNames, Integer.MAX_VALUE);
          writeToResolvedCache(_cacheGigaDir + File.separator + corefile, docsNames, docsEntities);
        }
      }
    }
  }
  
  /**
   * Main function that processes all stories in the given data object, and returns a list of those
   * stories, each story is a list of TextEntity objects. The documents in the data object are
   * put into the given docsNames list destructively.
   * @param data The stories you want to convert into entity lists.
   * @param docsNames An empty list that will be filled with the story names in your data object.
   * @return A list of entity lists. The list is as long as the number of stories given.
   * @param n The first n documents are read.
   */
  public List<List<TextEntity>> getEntityList(ProcessedData data, List<String> docsNames, int n) {
    IDFMap domainIDF = BaselineBestVerb.computeIDF(data, null, _wordnet);
    data.reset();
    data.nextStory();

    List<List<TextEntity>> docsEntities = new ArrayList<List<TextEntity>>();
    int xx = 0;

    while( data.getParseStrings() != null ) {
    	if( xx >= n ) break;

    	//      if( data.currentStory().contains("20061007") ) {
    	// Get the tokens and dependencies from this file.
    	List<TextEntity> entities = getEntityListCurrentDoc(data);

    	// Debugging output.
    	System.out.println("**Entities**\t" + data.currentStory() + "\t" + data.currentDoc());
    	System.out.println("\t" + entities);
    	System.out.println();

    	// Advance to next story.
    	docsEntities.add(entities);
    	docsNames.add(data.currentStory());
    	//      }
    	data.nextStory();
    	xx++;
    }

    if( debug )
      for( List<TextEntity> doc : docsEntities )
        for( TextEntity entity : doc ) System.out.println("..> " + entity);

    removeLowOccurringMentions(docsEntities);

    if( debug )
      for( List<TextEntity> doc : docsEntities )
        for( TextEntity entity : doc ) System.out.println("__> " + entity);

    removeLowIDFChange(docsEntities, generalIDF, domainIDF);

    if( debug )
      for( List<TextEntity> doc : docsEntities )
        for( TextEntity entity : doc ) System.out.println("==> " + entity);

    return docsEntities;
  }

  /**
   * Extract all entities from the given documents, and create their list of entity mentions.
   * This returns a list of all entities, with their mentions represented by a TextEntity
   * object that simply contains the head word and the dependency relation in which it is a dependent.
   * @param data
   * @return
   */
  public List<TextEntity> getEntityListCurrentDoc(ProcessedData data) {
    Map<Integer,TextEntity> idToEntity = new HashMap<Integer,TextEntity>();

    List<Tree> trees = TreeOperator.stringsToTrees(data.getParseStrings());
    List<List<TypedDependency>> alldeps = data.getDependencies();
    List<NERSpan> ners = data.getNER();

    if( trees.size() != alldeps.size() ) {
      System.out.println("Tree/Dep size no match in " + data.currentStory() + "(" + trees.size() + " " + alldeps.size());
    }
    
    // Add NER labels to the entity mentions.
    Collection<EntityMention> mentions = data.getEntities();
    addNERToEntities(mentions, ners);

    // Put the mentions in order of their sentences.
    List<EntityMention>[] mentionsBySentence = new ArrayList[trees.size()];
    for( EntityMention mention : mentions ) {
      if( mention.sid() > trees.size() ) {
        System.out.println("doc: " + data.currentStory());
        System.out.println("mention: " + mention);
        System.out.println("num trees: " + trees.size());        
      }
      if( mentionsBySentence[mention.sid()-1] == null )
        mentionsBySentence[mention.sid()-1] = new ArrayList<EntityMention>();
      mentionsBySentence[mention.sid()-1].add(mention);
    }

    // Remove duplicate entity mentions ("the outcome of the fighting" and "the fighting" both have "fighting" as the rightmost)
    // Proper head word detection would fix this, but no time...
    removeDuplicateMentions(mentionsBySentence);

    // Step through the sentences. Step through the entity mentions in each sentence.
    int sentid = 0;
    for( Tree tree : trees ) {
      if( mentionsBySentence[sentid] != null ) {
        List<String> leaves = TreeOperator.stringLeavesFromTree(tree);
        Collection<TypedDependency> sentdeps = alldeps.get(sentid);
        // Each token index has a list of dependencies in which that index was the dependent.
        List<List<String>> sortedDeps = sortDependenciesByDependent(sentdeps, tree);
        
        for( EntityMention mention : mentionsBySentence[sentid] ) {
          String leaf = leaves.get(mention.end()-1);
          List<String> deps = sortedDeps.get(mention.end()-1);
          NERSpan.TYPE ner = mention.namedEntity();

          // Token index starts at 1 for normalizing.
          String leaflemma = normalizeLeaf(leaf, mention.end(), tree);

          if( leaflemma.matches("^\\d+$") )
            leaflemma = intToDate(leaflemma, mention.end(), tree);

          if( deps != null ) {
            for( String dep : deps ) {
            	String olddep = dep;
            	dep = normalizeDep(dep);
            	// Create the entity if this is the first mention.
            	if( !idToEntity.containsKey(mention.entityID()) )
            		idToEntity.put(mention.entityID(), new TextEntity());
            	// Add the mention to the entity's list.
            	idToEntity.get(mention.entityID()).addMention(leaf, leaflemma, dep, ner);
            }
          }
        }
      }
      sentid++;
    }

    // Build the list object that holds all the entity objects.
    List<TextEntity> entities = new ArrayList<TextEntity>();
    for( Integer id : idToEntity.keySet() )
    	entities.add(idToEntity.get(id));

    // Set the core mentions for each entity.
    setCoreEntityMentions(entities);

    // Set the top-level word classes that this entity may belong to.
    setTopLevelAttributes(entities);

    if( debug )
    	for( TextEntity entity : entities )
    		System.out.println("--> " + entity);

    // Remove mentions that are reporting verbs.
    removeReportingMentions(entities);
    removeCommonMentions(entities);

    if( debug )
      for( TextEntity entity : entities )
        System.out.println("**> " + entity);

    // Remove entities with one mention that are pronouns.
    removePronounEntities(entities);

    if( debug )
      for( TextEntity entity : entities )
        System.out.println("!!> " + entity);

    // Remove entities with one mention that are just numbers (not years).
    removeNumberEntities(entities);

    // Remove any entity with only one mention.
    //    removeSingleMentionEntities(entities);
    if( debug )
      for( TextEntity entity : entities )
        System.out.println(">>> " + entity);


    return entities;
  }


  public Pair<List<String>,List<List<TextEntity>>> getCachedEntityList(String path) {
    return getResolvedCachedEntityList(createCachePath(path));
  }
  
  /**
   * This loads the data from a cached file.
   * @param path
   * @return
   */
  public Pair<List<String>,List<List<TextEntity>>> getResolvedCachedEntityList(String cachePath) {
    System.out.println("Reading from cache: " + cachePath);

    if( Directory.fileExists(cachePath) ) {
      try {
        BufferedReader in = new BufferedReader(new FileReader(cachePath));
        String line = in.readLine();

        // Get the doc names from the first line.
        String[] names = line.split("\t");
        List<String> docnames = new ArrayList<String>();
        for( String name : names ) docnames.add(name);

        // Now read the entity names.
        List<List<TextEntity>> docentities = new ArrayList<List<TextEntity>>();
        List<TextEntity> current = null;
        line = in.readLine();
        while( line != null ) {
          // New document
          if( line.startsWith("DOC") ) {
            if( current != null ) docentities.add(current);
            current = new ArrayList<TextEntity>();
          }
          else current.add(TextEntity.fromFullString(line));
          line = in.readLine();
        }
        // Add the final doc, the remainder.
        if( current != null ) docentities.add(current);

        // Return both the doc names and the entities per doc.
        return new Pair<List<String>,List<List<TextEntity>>>(docnames, docentities);
      } catch( Exception ex ) { 
        System.err.println("Error opening cache file: " + cachePath);
        ex.printStackTrace();
      }
    }
    else System.out.println("Cache file not found.");

    return null;
  }

  /**
   * Read the entities from a single document inside the given file. The file is assumed
   * to be moved to an offset that starts at a document. We read entities until the next
   * document is reached or end of file.
   * @param in An already opened file, offset moved to the doc we want.
   * @return The entities in the doc at the given offset.
   */
  public static List<TextEntity> readSingleDocRandomAccess(RandomAccessFile in) {
    try {
      // Read the line with the document name: "DOC APW_ENG_19990304.1234" (ignore it)
      String line = in.readLine();

      // Now read the entities, one per line.
      List<TextEntity> current = new ArrayList<TextEntity>();
      while( (line = in.readLine()) != null ) {
        // New document, stop and return.
        if( line.startsWith("DOC ") )
          return current;
        else 
          current.add(TextEntity.fromFullString(line));
      }
      // Hit the end of the file, return what we have.
      return current;
    } catch( Exception ex ) { 
      System.err.println("Error reading random access file.");
      ex.printStackTrace();
    }
    return null;
  }

  /**
   * Given the path to a file, this creates a new path to a cache directory that contains
   * the cache based on the given file.
   * @param origFile A file or a directory path.
   * @return A new file path, the cache equivalent location to the given path.
   */
  private String createCachePath(String origFile) {
  	String separator = File.separator;
  	if( separator.equals("\\") ) separator = "\\\\";  		

  	// Create the cache directory if it doesn't yet exist.
  	if( !(new File(_cacheDir).exists()) )
  		(new File(_cacheDir)).mkdir();
  	
  	System.out.println("cache path = " + origFile + " (separator=" + File.separator + ")");
    // Remove leading periods.
    while( origFile.charAt(0) == '.' || origFile.charAt(0) == File.separatorChar )
      origFile = origFile.substring(1);
    // Remove trailing periods.
    while( origFile.charAt(origFile.length()-1) == '.' || origFile.charAt(origFile.length()-1) == File.separatorChar )
      origFile = origFile.substring(0,origFile.length()-1);
    // Replace all double slashes
    origFile = origFile.replaceAll(separator + separator, "-");
    // Replace all single slashes
    origFile = origFile.replaceAll(separator, "-");
    origFile = origFile.replaceAll("/", "-");
    origFile = origFile.replaceAll("--+", "-");
    // Replace any other periods.
    origFile = origFile.replaceAll("\\.", "_");
    
    return _cacheDir + File.separator + origFile;
  }


  /**
   * Call this if you have a file path, and you want to create a cache path based on it.
   * This converts the given path to a cache directory and a simplified cache file name.
   */
  public void writeToCache(String path, List<String> docnames, List<List<TextEntity>> docEntities) {
    writeToResolvedCache(createCachePath(path), docnames, docEntities);
  }
  
  /**
   * Write all the entities to a single file.
   * The first line is the list of document names, separated by tabs.
   * Then each new doc starts with "DOC <docname>" and there is one entity per line.
   * @param cachepath Full path to a file to create and write to.
   * @param docnames List of document names.
   * @param docEntities List of entities per document, must align with docnames.
   */
  public void writeToResolvedCache(String cachepath, List<String> docnames, List<List<TextEntity>> docEntities) {
    System.out.println("Writing to cache: " + cachepath);

    try {
      PrintWriter writer = new PrintWriter(new BufferedWriter(new FileWriter(cachepath)));
      for( String name : docnames )
        writer.print(name + "\t");
      writer.println();

      int ii = 0;
      for( List<TextEntity> onedoc : docEntities ) {
        writer.print("DOC " + docnames.get(ii++) + "\n");
        for( TextEntity entity : onedoc ) {
          writer.print(entity.toFullString() + "\n");
        }
      }
      writer.close();
    } catch( Exception ex ) { ex.printStackTrace(); }
  }

  /**
   * Sometimes the coref system has two mentions that end on the same token, usually
   * a complex noun phrase subsumes the other. We want to keep the shorter one.
   * @param mentionsBySentence Array of mentions, each cell is a sentence and its mentions.
   */
  private void removeDuplicateMentions(List<EntityMention>[] mentionsBySentence) {
    for( int ii = 0; ii < mentionsBySentence.length; ii++ ) {
      List<EntityMention> mentions = mentionsBySentence[ii];
      Map<Integer,EntityMention> seenIndices = new HashMap<Integer,EntityMention>();
      Set<EntityMention> remove = new HashSet<EntityMention>();

      if( mentions != null ) {
        for( EntityMention current : mentions ) {
          if( !seenIndices.containsKey(current.end()) )
            seenIndices.put(current.end(), current);
          // Two mentions with same rightmost position in the sentence!
          else {
            EntityMention old = seenIndices.get(current.end());
            // Keep the shorter mention (assume the longer one is needlessly long)
            if( old.end() - old.start() > current.end() - current.start() ) {
              seenIndices.remove(old);
              seenIndices.put(current.end(), current);
              remove.add(old);
            } 
            else remove.add(current);
          }
        }
      }

      // Remove the duplicates.
      for( EntityMention mention : remove ) {
        mentions.remove(mention);
        if( debug ) System.out.println("Removing duplicate mention: " + mention);
      }
    }
  }


  /**
   * Remove all mentions from all entities that are arguments of reporting verbs.
   * This function also removes entities if they result in zero mentions remaining. 
   * @param entities List of entities from which to remove mentions.
   */
  private void removeCommonMentions(List<TextEntity> entities) {
    for( TextEntity entity : entities )
      removeCommonMentions(entity);
    // Delete entities without any remaining mentions.
    removeEntitiesWithNMentions(entities, 0);
  }

  /**
   * Remove all mentions that are arguments of reporting verbs.
   * This may result in an entity with zero mentions.
   * @param entity The entity with mentions.
   */
  private void removeCommonMentions(TextEntity entity) {
    for( int ii = 0; ii < entity.numMentions(); ii++ ) {
      String dep = entity.getMentionDependency(ii);
      String verb = dep.substring(dep.indexOf("--")+2);
      // If the mention is an argument of a reporting verb, remove.
      if( isCommonVerbLemma(verb) ) {
        if( debug ) System.out.println("Removing common verb mention: " + verb + "/" + dep + " for token " + entity.getMentionToken(ii));
        entity.removeMention(ii);
        // Recurse.
        removeCommonMentions(entity);
        return;
      }
    }
  }

  /**
   * Remove all mentions from all entities that are arguments of reporting verbs.
   * This function also removes entities if they result in zero mentions remaining. 
   * @param entities List of entities from which to remove mentions.
   */
  private void removeReportingMentions(List<TextEntity> entities) {
    for( TextEntity entity : entities )
      removeReportingMentions(entity);
    // Delete entities without any remaining mentions.
    removeEntitiesWithNMentions(entities, 0);
  }

  /**
   * Remove all mentions that are arguments of reporting verbs.
   * This may result in an entity with zero mentions.
   * @param entity The entity with mentions.
   */
  private void removeReportingMentions(TextEntity entity) {
    for( int ii = 0; ii < entity.numMentions(); ii++ ) {
      String dep = entity.getMentionDependency(ii);
      String verb = dep.substring(dep.indexOf("--")+2);
      // If the mention is an argument of a reporting verb, remove.
      if( isReportingVerbLemma(verb) ) {
        if( debug ) System.out.println("Removing reporting mention: " + verb + "/" + dep + " for token " + entity.getMentionToken(ii));
        entity.removeMention(ii);
        // Recurse.
        removeReportingMentions(entity);
        return;
      }
    }
  }

  /**
   * Remove all entities that have a single mention only.
   * @param entities List of entity objects.
   */
  private void removeEntitiesWithNMentions(List<TextEntity> entities, int n) {
    List<TextEntity> remove = new ArrayList<TextEntity>();
    for( TextEntity entity : entities ) {
      if( entity.numMentions() <= n )
        remove.add(entity);
    }
    for( TextEntity entity : remove )
      entities.remove(entity);    
  }

  /**
   * Remove all entities that have a core token which is a pronoun.
   * @param entities
   */
  private void removePronounEntities(List<TextEntity> entities) {
    List<TextEntity> remove = new ArrayList<TextEntity>();
    for( TextEntity entity : entities ) {
      //      if( entity.numMentions() == 1 )
      if( Ling.isPersonPronoun(entity.getCoreToken()) || Ling.isInanimatePronoun(entity.getCoreToken()) )
        remove.add(entity);
    }
    for( TextEntity entity : remove )
      entities.remove(entity);
  }

  /**
   * Remove all entities that have a core token which is a pronoun.
   * @param entities
   */
  private void removeNumberEntities(List<TextEntity> entities) {
    List<TextEntity> remove = new ArrayList<TextEntity>();
    for( TextEntity entity : entities ) {
      if( entity.numMentions() == 1 ) {
        String token = entity.getMentionToken(0);
        if( token.matches("^\\d+$") ) {
          Integer value = Integer.parseInt(token);
          if( value < 1900 || value > 2020 )
            remove.add(entity);
        }
      }
    }
    for( TextEntity entity : remove ) {
      System.out.println("Number Removing entity: " + entity);
      entities.remove(entity);
    }
  }
  
  private void removeLowOccurringMentions(List<List<TextEntity>> allentities) {
    // Count all mentions.
    Counter<String> counter = new ClassicCounter<String>();
    for( List<TextEntity> docentities : allentities ) {
      for( TextEntity entity : docentities ) {
        for( int ii = 0; ii < entity.numMentions(); ii++ ) {
          String dep = entity.getMentionDependency(ii);
          counter.incrementCount(dep);
        }
      }
    }
    if( debug ) {
      System.out.println("DEP COUNTS");
      for( String key : counter.keySet() ) System.out.println("..." + key + " " + counter.getCount(key));
    }

    // Remove mentions
    for( List<TextEntity> docentities : allentities ) {
      for( TextEntity entity : docentities ) {
        boolean removed = true;
        while( removed ) {
          removed = false;
          for( int ii = 0; ii < entity.numMentions(); ii++ ) {
            String dep = entity.getMentionDependency(ii);
            double count = counter.getCount(dep);
            if( count < _minDepCounts ) {
              if( debug ) System.out.println("Removing rare mention " + entity.getMentionToken(ii) + " " + dep);
              entity.removeMention(ii);
              removed = true;
              break;
            }
          }
        }
      }
      // Delete entities without any remaining mentions.
      removeEntitiesWithNMentions(docentities, 0);
    }
  }
  
  private void removeLowIDFChange(List<List<TextEntity>> allentities, IDFMap generalIDF, IDFMap domainIDF) {
//    DomainVerbDetector verbDetector = new DomainVerbDetector(generalIDF, domainIDF);
//    double filatovaIDFCutoff = 2.0;
//    double scoreCutoff = 0.0001;
    //    System.out.println("General IDF Cutoff = " + filatovaIDFCutoff);
      //    System.out.println("Domain Cutoff = " + scoreCutoff);

    // Sum total frequencies of all seen words.
    // int domainTotalOccurrences = 0;
    // for( String word : domainIDF.getWords() ) {
    //   if( generalIDF.get(word) >= filatovaIDFCutoff )
    //     domainTotalOccurrences += domainIDF.getFrequency(word);
    // }

    // Grabs only the top 200 verbs.
    List<String> topVerbs = getTopVerbs(generalIDF, domainIDF, 1000);
    System.out.println("Top verbs: " + topVerbs);
    
    // Remove mentions
    for( List<TextEntity> docentities : allentities ) {
      for( TextEntity entity : docentities ) {
        boolean removed = true;
        while( removed ) {
          removed = false;
          for( int ii = 0; ii < entity.numMentions(); ii++ ) {
            String dep = entity.getMentionDependency(ii);
            String verb = dep.substring(dep.indexOf("--")+2); // kidnap

            if( !topVerbs.contains("v-"+verb) && !topVerbs.contains("n-"+verb) ) removed = true;
            
//            verb = "v-" + verb;
//            System.out.println("verb=" + verb + "\t" + generalIDF.get(verb));
//            if( generalIDF.get(verb) < filatovaIDFCutoff )
//              removed = true;
//            else {
//              // probability of the word
//              double score = (double)domainIDF.getFrequency(verb) / (double)domainTotalOccurrences;
//              // multiplied by an IDF score
//              score *= (double)domainIDF.getDocCount(verb) / (double)domainIDF.numDocs();
//              System.out.println("IDF Scored " + verb + "\t" + score);
//              if( score < scoreCutoff )
//                removed = true;
//            }
            
            if( removed ) {
            	System.out.println("Removing mention with low IDF change verb: " + dep + "-" + verb);
              entity.removeMention(ii);
              break;
            }
          }
        }
      }
      // Delete entities without any remaining mentions.
      removeEntitiesWithNMentions(docentities, 0);
    }
  }

  /**
   * Get the top n words by the ratio of general to domain IDF scores.
   * @param generalIDF
   * @param domainIDF
   * @param n The number of top words to return in order.
   * @return An ordered list of the top words.
   */
  private List<String> getTopVerbs(IDFMap generalIDF, IDFMap domainIDF, int n) {
    Counter<String> wordscores = new ClassicCounter<String>();    
    for( String word : domainIDF.getWords() ) {
      if( word.startsWith("v-") || (word.startsWith("n-") && _wordnet.isNounEvent(word.substring(2))) ) {
      	System.out.println("word: " + word + "\tgeneralIDF=" + generalIDF.get(word) + "\tdomainDocCount=" + domainIDF.getDocCount(word));
        if( (generalIDF.get(word) >= 0.2 || !generalIDF.contains(word)) && 
        		(domainIDF.getDocCount(word) > _minDocCounts || domainIDF.getDocCount(word) > domainIDF.numDocs()/5) ) {

          // ratio
          double score = generalIDF.get(word) / domainIDF.get(word);

          // filtaova : probability of word, multiplied by IDF score
//          score = (double)domainIDF.getFrequency(word) / (double)domainIDF.totalCorpusCount();
//          score *= (double)domainIDF.getDocCount(word) / (double)domainIDF.numDocs();

          wordscores.incrementCount(word, score);
          System.out.println("scored: " + word + "\t" + score);
        }
      }
    }
    
    List<String> sortedWords = Util.sortCounterKeys(wordscores);
    List<String> topWords = new ArrayList<String>();
    for( int ii = 0; ii < n && ii < sortedWords.size(); ii++ ) {
      topWords.add(sortedWords.get(ii));
      System.out.println("top: " + sortedWords.get(ii) + "\t" + wordscores.getCount(sortedWords.get(ii)));
    }
    return topWords;
  }
  
  /**
   * The TextEntitly objects should have NER labels already assigned per mention from
   * the NER system. Thus function looks at those, but also the top-level WordNet synsets
   * of each entity's core token to find possible attributes.
   * The result is an entity with binary attributes of PERSON, PHYSOBJECT, LOCATION, EVENT, and OTHER.
   * 
   * @param entity The entity itself with NER labels already assigned to it.
   */
  private void setTopLevelAttributes(List<TextEntity> entities) {
    for( TextEntity entity : entities ) {
      Set<TextEntity.TYPE> types = validEntityRoleTypes(entity.ners, entity.getCoreToken());
      entity.setEntityTypes(types);
    }
  }

  private void setCoreEntityMentions(List<TextEntity> entities) {
    for( TextEntity entity : entities ) setCoreEntityMention(entity);
  }

  /**
   * Determines which mention of the entity contains the key phrase.
   * This is done solely by mention string length.
   * @param entity
   */
  private void setCoreEntityMention(TextEntity entity) {
    String bestToken = null;
    int mentioni = 0, besti = 0;
    for( String token : entity.tokens ) {
      if( bestToken == null ) {
        bestToken = token;
        besti = mentioni;
      }
      // Take the longer token as the better mention.
      else if( token.length() > bestToken.length() ) {
        // Don't set the longer token if it is a pronoun.
        // (yes, the current best might be a pronoun, but then it doesn't matter which we choose)
        if( !Ling.isPersonPronoun(token) && !Ling.isInanimatePronoun(token) && 
            !Ling.isAbstractPerson(token) && !Ling.isObjectReference(token) ) {
          bestToken = token;
          besti = mentioni;
        }
      }
      mentioni++;
    }

    entity.setCoreToken(bestToken, entity.rawTokens.get(besti));

    if( bestToken.matches("^\\d+$") ) {
      if( debug ) System.out.println("Number turned from " + bestToken);
      entity.setCoreToken("NUMBER", entity.rawTokens.get(besti));
    }

    // Set this entity's token to be an NER label if it has one.
    if( _changeTokensToNELabel ) {
      NERSpan.TYPE bestNER = entity.ners.get(besti); 
      if( bestNER == NERSpan.TYPE.PERSON )
        entity.setCoreToken("PERSON", entity.rawTokens.get(besti));
      else if( bestNER == NERSpan.TYPE.ORGANIZATION )
        entity.setCoreToken("ORG", entity.rawTokens.get(besti));
    }
  }

  /**
   * Markup the entity mentions with the NER label if there is one, based on the
   * rightmost word in the entity mention.
   * @param mentions All the entity mentions in a document.
   * @param ners All the NER spans in a document. 
   */
  private void addNERToEntities(Collection<EntityMention> mentions, List<NERSpan> ners) {
    //    System.out.println("addNERToEntities()");
    for( EntityMention mention : mentions ) {
      int rightmost = mention.end();
      for( NERSpan span : ners ) {
        if( span.sid() == mention.sid()-1 && span.start() <= rightmost && span.end() > rightmost ) {
          //          System.out.println("Adding NER " + span + " to mention " + entity);
          mention.setNamedEntity(span.type());
          break;
        }
      }
    }
  }

  /**
   * Given an entity (and its NE labels), add to the NE labels based on a lookup of its
   * head word in wordnet to see if we can determine if the entity is a person, location,
   * or other role type. We return the set of possible types.
   * @param mentionNELabels The list of all NE labels assigned to mentions by the NER system.
   * @param mainDescriptor The main string description for the entity.
   */
  private Set<TextEntity.TYPE> validEntityRoleTypes(List<NERSpan.TYPE> mentionNELabels, String mainDescriptor) {
    Set<TextEntity.TYPE> validEntityTypes = new HashSet<TextEntity.TYPE>();

    // Get rid of the NONE labels.
    Set<NERSpan.TYPE> labeledNEs = new HashSet<NERSpan.TYPE>();
    for( NERSpan.TYPE ne : mentionNELabels ) {
      if( ne != NERSpan.TYPE.NONE )
        labeledNEs.add(ne);
    }

    String key = mainDescriptor;
    int space = mainDescriptor.lastIndexOf(' ');
    if( space > -1 ) key = mainDescriptor.substring(space+1);

    // TODO: temporary check
    if( key.equalsIgnoreCase("person") || key.equalsIgnoreCase("people") ) {
      validEntityTypes.add(TextEntity.TYPE.PERSON);
      return validEntityTypes;
    }

    if( key.equalsIgnoreCase("TIMEDATE") ) {
      validEntityTypes.add(TextEntity.TYPE.TIME);
      return validEntityTypes;
    }

    if( Ling.isPersonPronoun(key) ) {
      validEntityTypes.add(TextEntity.TYPE.PERSON);
      return validEntityTypes;
    }

    //    System.out.println("valid lookup key " + key + " ners " + labeledNEs);
    //    System.out.println("  isPerson " + SlotTypeCache.isPerson(key, _wordnet));
    //    System.out.println("  isPhysObj " + SlotTypeCache.isPhysObject(key, _wordnet));
    //    System.out.println("  isLocation " + SlotTypeCache.isLocation(key, _wordnet));
    //    System.out.println("  isEvent " + SlotTypeCache.isEvent(key, _wordnet));
    //    System.out.println("  isUnk " + _wordnet.isUnknown(key));
    //    System.out.println("  isOther " + SlotTypeCache.isOther(key, _wordnet));

    if( labeledNEs.contains(NERSpan.TYPE.LOCATION) || SlotTypeCache.isLocation(key, _wordnet) )
      validEntityTypes.add(TextEntity.TYPE.LOCATION);
    if( labeledNEs.contains(NERSpan.TYPE.PERSON) || SlotTypeCache.isPerson(key, _wordnet) )
      validEntityTypes.add(TextEntity.TYPE.PERSON);
    if( labeledNEs.contains(NERSpan.TYPE.ORGANIZATION) )
      validEntityTypes.add(TextEntity.TYPE.ORG);

    // Don't label things events if they had NER tags.
    if( labeledNEs.size() == 0 && SlotTypeCache.isEvent(key, _wordnet) )
      validEntityTypes.add(TextEntity.TYPE.EVENT);

    // Don't label things as physical objects if they had NER tags.
    if( labeledNEs.size() == 0 && SlotTypeCache.isPhysObject(key, _wordnet) )
      validEntityTypes.add(TextEntity.TYPE.PHYSOBJECT);

    // If no attributes yet, and the key contains non a-z characters, just label it OTHER and return.
    if( !key.matches("^[A-Za-z]+$") ) {
      validEntityTypes.add(TextEntity.TYPE.OTHER);
      return validEntityTypes;
    }

    // Unknown words could be people or locations.
    if( validEntityTypes.size() == 0 && _wordnet.isUnknown(key) ) {
      validEntityTypes.add(TextEntity.TYPE.PERSON);
      validEntityTypes.add(TextEntity.TYPE.ORG);
      validEntityTypes.add(TextEntity.TYPE.LOCATION);
      if( debug ) System.out.println("Totally unknown word: " + key + " (now listed as PERSON or ORG or LOCATION)");
    }    

    // Don't label things events if they had NER tags.
    if( labeledNEs.size() == 0 && SlotTypeCache.isTime(key, _wordnet) )
      validEntityTypes.add(TextEntity.TYPE.TIME);

    //    // Physical objects (non-people) are OTHER.
    //    if( _wordnet.isUnknown(key) || _wordnet.isPhysicalObject(key) || (_wordnet.isMaterial(key) && !_wordnet.isTime(key)) )
    //      validEntityTypes.add(TextEntity.TYPE.OTHER);
    //
    //    // Other is also basically anything that isn't a person or location...
    //    else if( SlotTypeCache.isOther(key, _wordnet) && !_wordnet.isTime(key) )
    //      validEntityTypes.add(TextEntity.TYPE.OTHER);

    if( validEntityTypes.size() == 0 )
      validEntityTypes.add(TextEntity.TYPE.OTHER);

    return validEntityTypes;
  }

  /**
   * @return The NE labels that tagged any of the entity's mentions.
   */
  private Set<NERSpan.TYPE> entityMentionNETypes(List<EntityMention> mentions) {
    Set<NERSpan.TYPE> types = new HashSet<NERSpan.TYPE>();
    for( EntityMention mention : mentions ) {
      if( mention.namedEntity() != NERSpan.TYPE.NONE )
        types.add(mention.namedEntity());
    }
    return types;
  }

  /**
   * Sometimes a digit is the day of a month, so we look for a time unit on either
   * side of the given digit in the sentence. If it exists, we return a new string
   * with both tokens appended;
   * @param token
   * @param index
   * @param tree
   * @return
   */
  private String intToDate(String token, int index, Tree tree) {
    if( token.matches("^\\d+$") ) {
//      System.out.println("intToDate " + token);

      String pre = (index > 1 ? TreeOperator.indexToToken(tree, index-1) : null);
      String pre2 = (index > 2 ? TreeOperator.indexToToken(tree, index-2) : null);
      String post = TreeOperator.indexToToken(tree, index+1);

      if( (pre != null && _wordnet.isTime(pre)) || 
          (post != null && _wordnet.isTime(post)) ) {
        if( debug ) System.out.println("Timedate cal rule matched: " + token);
        return "TIMEDATE";
      }

      // e.g. 0800
      if( token.matches("^0\\d\\d\\d$") ) {
        if( debug ) System.out.println("Timedate 0 rule matched: " + token);
        return "TIMEDATE";
      }

      // e.g. 1700 (5pm)
      if( token.length() < 5 && Integer.parseInt(token) <= 2400 ) {
        if( pre != null && (pre.equalsIgnoreCase("at") || pre.equalsIgnoreCase("at")) ) {
          if( debug ) System.out.println("Timedate 'at' rule matched: " + token);
          return "TIMEDATE";
        }
      }        

    }
    return token;
  }

  /**
   * Lemmatize a token, given its position in a parse tree.
   * @param leaf The original token in the sentence.
   * @param leafIndex The index of the token in the sentence. First word starts at 1, not 0.
   * @param tree The parse tree of the sentence.
   * @return A lemmatized leaf, or the original if lemmatization fails.
   */
  public String normalizeLeaf(String leaf, int leafIndex, Tree tree) {
    leaf = leaf.toLowerCase();

    Tree subtree = TreeOperator.indexToSubtree(tree, leafIndex);
    if( subtree == null ) {
      return leaf;
    }
    String posTag = subtree.label().value();
    String lemma = _wordnet.lemmatizeTaggedWord(leaf, posTag);
//    System.out.println("normalizeLeaf:\t" + leaf + "\t" + posTag + "\t" + lemma);
    return lemma;
  }

  public String normalizeDep(String dep) {
    if( dep == null )
      return null;
    else if( dep.contains("nsubjpass") ) 
      return dep.replaceFirst("nsubjpass--", "dobj--");
    else if( dep.contains("agent--") ) 
      return dep.replaceFirst("agent--", "nsubj--");
    else if( dep.contains("xsubj--") )
      return dep.replaceFirst("xsubj--", "nsubj--");
    else return dep;
  }

  public boolean depIsImportant(String dep) {
    if( dep.startsWith("nsubj") || dep.startsWith("agent") || dep.startsWith("dobj") )
      return true;
    return false;
  }

  public boolean leafIsImportant(String leaf) {
    if( leaf.equals("this") ||
        Ling.isPersonPronoun(leaf) || Ling.isInanimatePronoun(leaf) )
      return false;
    else
      return true;
  }

  /**
   * Put all of the tokens in the document in order in one long list.
   * @param data The parsed document's file path.
   * @return A list of tokens.
   */
  public List<String> getTokens(ProcessedData data) {
    List<String> tokens = new ArrayList<String>();

    List<String> parseStrings = data.getParseStrings();
    for( String parseString : parseStrings ) {
      Tree tree = TreeOperator.stringToTree(parseString, _tf);
      List<String> leaves = TreeOperator.stringLeavesFromTree(tree);
      for( String leaf : leaves )
        tokens.add(leaf.toLowerCase());
    }

    return tokens;
  }

  /**
   * This looks at the token indices, and finds a dependency relation for each index.
   * It then stringifies these, and returns them in order of index from low to high.
   * The list is as long as the sentence, with null entries for token indices that had
   * no such relation.
   * @param deps The dependencies of the sentence.
   * @param tokens The tokens of the sentence, pulled from the phrase structure tree.
   * @param tree  The parse tree of the sentence.
   * @return
   */
  private List<List<String>> sortDependenciesByDependent(Collection<TypedDependency> deps, Tree tree) {
    List<String> tokens = TreeOperator.stringLeavesFromTree(tree);
    int numTokens = tokens.size();

    List<List<String>> ordered = new ArrayList<List<String>>();
    for( int ii = 0; ii < numTokens; ii++ ) {
      // Get the relation. Dependency tokens are indexed from 1, not 0.
      List<String> relations = getRelationForDependent(deps, ii+1, tokens.get(ii), tree);
      ordered.add(relations);
    }

    // Add nulls till the sentence ends.
    while( ordered.size() < numTokens )  ordered.add(null);

    return ordered;
  }

  /**
   * Given a token index in the sentence, find the dependency relation that has it as a dependent.
   * Generate a string representing the relation: reln--governor
   *
   *
   * TODO: some tokens might be the dependent of multiple governors (conjunctions, in particular).
   *       This function does nothing for those and only choose the first one!!!  Do something else in the future??
   *       e.g. <D>dobj laying-20 weapons-23</D>
   *            <D>nsubjpass destroyed-29 weapons-23</D>
   *       e.g. Saad, announced, today, that, a, group, of, Colombians, dressed, as, military, men, on, 28, April, Kidnapped, U.S., citizen, Scott, heyndal, ,, killed, a, Colombian, ,, and, wounded, an, Ecuadorean...
   *            
   * @param deps  The sentence's dependency graph.
   * @param index The target token's index.
   * @param token The token itself (for sanity checking the index).
   * @param tree  The parse tree of the sentence.
   * @return A String "feature" of the dependency relation: reln--governor
   */
  private List<String> getRelationForDependent(Collection<TypedDependency> deps, int index, String token, Tree tree) {
    List<String> bestDeps = null;
    for( TypedDependency dep : deps ) {
      if( dep.dep().index() == index ) {
        // Sanity check that the dependency's token is the parse tree's token.
        // Rare: 1979 is the token, (1979) is the dependency token.
        if( !dep.dep().nodeString().equals(token) && !dep.dep().nodeString().contains(token) ) {
          System.out.println("Token positions don't line up. token=" + token + " and dep=" + dep);
          System.exit(-1);
        }

        // Lemmatize and lowercase the governor.
        String governor = dep.gov().nodeString();
        governor = normalizeLeaf(governor, dep.gov().index(), tree);

        // Look for particles.
        Map<Integer,String> particles = Ling.particlesInSentence(deps);
        if( particles.containsKey(dep.gov().index()) ) {
          governor += "_" + particles.get(dep.gov().index());
          //          System.out.println("PARTICLE: " + governor);
        }          

        // Build the feature string and return it.
        if( bestDeps == null ) {
          bestDeps = new ArrayList<String>();
          bestDeps.add(dep.reln() + "--" + governor);
        }
        // If there is a tmod, just add this one tmod and don't add anything else.
        else if( dep.reln().toString().equals("tmod") ) {
          bestDeps.clear();
          bestDeps.add("tmod--" + governor);
          break;
        }
        else {
          bestDeps.add(dep.reln() + "--" + governor);
        }
      }
    }
    return bestDeps;
  }

  /**
   * True if the given token is the lemma of a reporting verb. Assumes lowercase input.
   * @param token The token to check.
   * @return True if a reporting verb, false otherwise.
   */
  public static boolean isReportingVerbLemma(String token) {
    if( token.equals("say") || token.equals("report") || token.equals("reply") || token.equals("tell") || 
    		token.equals("talk") || token.equals("add") )
      return true;
    else
      return false;
  }

  /**
   * True if the given token is the lemma of a reporting verb. Assumes lowercase input.
   * @param token The token to check.
   * @return True if a reporting verb, false otherwise.
   */
  public static boolean isCommonVerbLemma(String token) {
    if( token.equals("be") || token.equals("have") || token.equals("do") )
      return true;
    else
      return false;
  }





  /**
   * Pull out a dependency for each token, and return the dependencies in order
   * of the tokens so they line up by index.
   * @param data The parsed document's file path.
   * @return A list of dependencies. null indicates the token had no parent in the graph. 
   */
  private List<String> getDependenciesPerToken(ProcessedData data) {
    List<String> tokens = new ArrayList<String>();
  
    List<Tree> trees = TreeOperator.stringsToTrees(data.getParseStrings());
    List<List<TypedDependency>> alldeps = data.getDependencies();
  
    int xx = 0;
    for( Collection<TypedDependency> sentdeps : alldeps ) {
      Tree tree = trees.get(xx++);
  
      List<List<String>> sortedDeps = sortDependenciesByDependent(sentdeps, tree);
      for( List<String> deps : sortedDeps )
        for( String dep : deps )
          tokens.add(dep.toLowerCase());
    }
  
    return tokens;
  }

  /**
   * Puts all of the tokens in the document in order in one long list.
   * Pairs up the dependency relation of each token in a pair object.
   * This function throws out tokens that have a null dep relation, so the number
   * of returned tokens (pairs) are less than the length of each document. 
   * @param data The parsed document's file path.
   * @return A list of pairs: <token, dependency>
   */
  public List<Pair<String,String>> getTokenDepPairs(ProcessedData data) {
    List<Pair<String,String>> tokenDeps = new ArrayList<Pair<String,String>>();

    List<Tree> trees = TreeOperator.stringsToTrees(data.getParseStrings());
    List<List<TypedDependency>> alldeps = data.getDependencies();

    int xx = 0;
    for( Tree tree : trees ) {
      Collection<TypedDependency> sentdeps = alldeps.get(xx);
      List<String> leaves = TreeOperator.stringLeavesFromTree(tree);

      List<List<String>> sortedDeps = sortDependenciesByDependent(sentdeps, tree);

      int yy = 0;
      for( String leaf : leaves ) {
        leaf = leaf.toLowerCase();
        //        if( leafIsImportant(leaf) ) {
        if( true ) {
          List<String> deps = sortedDeps.get(yy);
          if( deps != null ) {
            for( String dep : deps ) {
              if( depIsImportant(dep) ) {
                dep = normalizeDep(dep);
                tokenDeps.add(new Pair<String,String>(leaf, dep.toLowerCase()));
              }
            }
          }
        }
        yy++;
      }
      xx++;
    }

    return tokenDeps;
  }
  
  /**
   * For pre-processing Gigaword only.
   */
  public static void main(String[] args) {
    String parseDir = null;
    String depDir = null;
    String entityDir = null;
    String nerDir = null;
    
    // data and lore
    if( Directory.fileExists("/home/nchamber/data/gigaword/charniak-based/apw_eng") ) {
      parseDir = "/home/nchamber/data/gigaword/charniak-based/apw_eng/phrase";
      depDir = "/home/nchamber/data/gigaword/charniak-based/apw_eng/deps";
      nerDir = "/home/nchamber/data/gigaword/charniak-based/apw_eng/ner";
      //entityDir = "/home/nchamber/data/gigaword/charniak-based/apw_eng/corefOpennlp";
      entityDir = "/home/nchamber/data/gigaword/charniak-based/apw_eng/corefStanford";
    }
    else System.out.println("ERROR: couldn't find data directories in IRDocuments. Hardcode it in the code.");
    
    DataSimplifier simp = new DataSimplifier();
    simp.simplifyGigaword(parseDir, depDir, entityDir, nerDir);
  }
}
