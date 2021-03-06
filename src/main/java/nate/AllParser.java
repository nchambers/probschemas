package nate;

import java.io.File;
import java.io.PrintWriter;
import java.io.StringWriter;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.Vector;

import nate.util.*;
import edu.stanford.nlp.dcoref.CorefChain;
import edu.stanford.nlp.dcoref.CorefChain.CorefMention;
import edu.stanford.nlp.dcoref.CorefCoreAnnotations.CorefChainAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.CoreAnnotations.NamedEntityTagAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TokensAnnotation;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.parser.lexparser.Options;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.trees.GrammaticalStructure;
import edu.stanford.nlp.trees.GrammaticalStructureFactory;
import edu.stanford.nlp.trees.PennTreebankLanguagePack;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.trees.TreeCoreAnnotations.TreeAnnotation;
import edu.stanford.nlp.trees.TreePrint;
import edu.stanford.nlp.trees.TreebankLanguagePack;
import edu.stanford.nlp.trees.TypedDependency;
import edu.stanford.nlp.util.CoreMap;


/**
 * Creates two files as output: parses and dependencies
 *
 * -output
 * Directory to create parsed and dependency files.
 *
 * -continue
 * Part of a file name of a file to start parsing again.
 * Usually it was interrupted and we need to continue.
 *
 * -input giga|muc|text
 * The type of text we are processing, Gigaword or Environment.
 */
public class AllParser {
  private static StanfordCoreNLP pipeline;
  
  public int MAX_SENTENCE_LENGTH = 400;
  String _dataPath = "";
  String _outputDir = ".";
  Options options;
  GrammaticalStructureFactory gsf;
  // One file name to continue parsing.
  // Usually the previous run was interrupted.
  String _continueFile = null; 
  boolean debug = false;
  
  public static final int GIGAWORD = 0;
  public static final int ENVIRO = 1;
  public static final int MUC = 2;
  public static final int TEXT = 3;
  
  private int _docType = TEXT;


  public AllParser(String[] args) {
    if( args.length < 2 ) {
      System.out.println("DirectoryParser [-output <dir>] -type giga|muc -input <text-directory>");
      System.exit(-1);
    }
    handleParameters(args);
    initLexResources();
  }

  private void handleParameters(String[] args) {
    HandleParameters params = new HandleParameters(args);

    if( params.hasFlag("-continue") ) {
      _continueFile = params.get("-continue");
      System.out.println("Continuing on file " + _continueFile);
    }
    if( params.hasFlag("-output") ) {
      _outputDir = params.get("-output");
    }
    if( params.hasFlag("-type") ) {
      _docType = docTypeToInt(params.get("-type"));
    }
    if( params.hasFlag("-input") ) {
      _dataPath = params.get("-input");
    }

    // Sanity check for input path.
    if( _dataPath == null || _dataPath.length() == 0 ) {
      System.err.println("ERROR: no input path given.");
      System.exit(1);      
    }

    System.out.println("Input type: " + _docType);
    System.out.println("Input path: " + _dataPath);
    System.out.println("Output dir: " + _outputDir);
  }
  
  public static int docTypeToInt(String str) {
    str = str.toLowerCase();
    if( str.startsWith("giga") )
      return GIGAWORD;
    else if( str.startsWith("env") )
      return ENVIRO;
    else if( str.startsWith("muc") )
      return MUC;
    else if( str.equals("text") )
      return TEXT;
    else {
      System.out.println("Unknown text input type: " + str);
      System.exit(1);
    }
    return -1;
  }

  private void initLexResources() {
    Properties props = new Properties();
    props.put("annotators", "tokenize, ssplit, pos, lemma, parse, ner, dcoref"); // most of these are dependencies for the main ones.
    pipeline = new StanfordCoreNLP(props);
    
    try {
      options = new Options();
      options.testOptions.verbose = true;
    } catch( Exception ex ) { ex.printStackTrace(); }

    // Dependency tree info
    TreebankLanguagePack tlp = new PennTreebankLanguagePack();
    gsf = tlp.grammaticalStructureFactory();
  }


  /**
   * @desc Parses input sentences and prints the parses to the given doc.
   * @param currentStory The name of the document these sentences are from.
   * @param currentStoryNum Unique ID for this document/story.
   * @param paragraphs Vector of strings of sentences
   * @param pdoc The current document we're printing to.
   * @param depdoc The document of dependencies that we're printing to.
   * @param corefdoc The document of coref chains that we're printing to.
   */
  private void analyzeSentences(String currentStory, int currentStoryNum, Vector<String> paragraphs, GigaDoc pdoc, GigaDoc depdoc, GigaDoc corefdoc, GigaDoc nerdoc) {
    int sid = 0;

    // Paragraphs may be multiple sentences
    String allsents = "";
    for( String fragment : paragraphs ) {
      allsents += fragment + " \n";
    }

    // Replace underscores (gigaword has underscores in many places commas should be)
    if( allsents.contains(" _ ") ) allsents = allsents.replaceAll(" _ ", " , ");

    if( debug ) System.out.println("NEW: sentences = " + allsents);
    Annotation document = new Annotation(allsents);
    
    try {
      pipeline.annotate(document);
    } catch( Exception ex ) {
      ex.printStackTrace();
      System.out.println("ERROR: skipping document due to Stanford CoreNLP failure.");
      return;
    }

    // Loop over the sentences.
    List<Tree> trees = new ArrayList<Tree>();
    List<CoreMap> sentences = document.get(CoreAnnotations.SentencesAnnotation.class);
    for( CoreMap sentence: sentences ) {
      if( debug ) System.out.println("sid = " + sid);
      if( debug ) System.out.println("sent = " + sentence);

      // PARSE TREE
      Tree tree = sentence.get(TreeAnnotation.class);
      trees.add(tree);
      if( debug ) System.out.println("tree = " + tree);
      // Build a StringWriter, print the tree to it, then save the string
      StringWriter treeStrWriter = new StringWriter();
      TreePrint tp = new TreePrint("penn");
      tp.printTree(tree, new PrintWriter(treeStrWriter,true));
      pdoc.addParse(treeStrWriter.toString());

      // DEPENDENCY GRAPH (old way of doing it)
      // Create the dependency tree - CAUTION: DESTRUCTIVE to parse tree
      try {
        GrammaticalStructure gs = gsf.newGrammaticalStructure(tree);
        //      Collection<TypedDependency> deps = gs.typedDependenciesCollapsed();
        Collection<TypedDependency> deps = gs.typedDependenciesCCprocessed(true);
        depdoc.addDependencies(deps, sid);
      } catch( Exception ex ) { 
        ex.printStackTrace();
        System.out.println("WARNING: dependency tree creation failed...adding null deps");
        depdoc.addDependencies(null, sid);
      }

      // This is the new way to make Dependency Graphs, but I can't shove it into my old data structure...
      //        SemanticGraph dependencies = sentence.get(CollapsedCCProcessedDependenciesAnnotation.class);
      //        System.out.println("deps = " + dependencies.getEdgeSet());
      //        for( SemanticGraphEdge edge : dependencies.edgeIterable() ) {
      //          System.out.println("\t" + edge);
      //          System.out.println("\tgov=" + edge.getGovernor() + "\trel=" + edge.getRelation() + "\tdep=" + edge.getDependent());
      //          TreeGraphNode node = new TreeGraphNode();
      //          TypedDependency tdep = new TypedDependency(edge.getRelation(), edge.getGovernor(), edge.getDependent());
      //        }        

      // NER
      String prev = null;
      int start = -1;
      int i = 1;
      for( CoreLabel token : sentence.get(TokensAnnotation.class) ) {
        String ne = token.get(NamedEntityTagAnnotation.class);
//        System.out.println("ne=" + ne + " prev=" + prev);
        // If no NE here, but there was a previous NE.
        if( ne.equals("O") && prev != null ) {
          NERSpan.TYPE type = nerStringToType(prev);
          if( type != null ) nerdoc.addNER(new NERSpan(type, sid, start, i-1));
//          System.out.println("Added1 " + (new NERSpan(type, sid, start, i-1)));
          prev = null;
          start = -1;
        }
        // If a different NE is next to the previous NE.
        else if( !ne.equals("O") && prev != null && !ne.equals(prev) ) {
          NERSpan.TYPE type = nerStringToType(prev);
          if( type != null ) nerdoc.addNER(new NERSpan(type, sid, start, i-1));
//          System.out.println("Added2 " + (new NERSpan(type, sid, start, i-1)));
          prev = ne;
          start = i;          
        }
        // Starting a new NE after some O's.
        else if( !ne.equals("O") && prev == null ) {
          prev = ne;
          start = i;          
        }
        // Else the same NE continues.
        else { }
//        System.out.println("ne=" + ne + " " + token);
        i++;
      }
//      System.exit(1);
      
      sid++;
    }
    
    // COREFERENCE
    Map<Integer, CorefChain> graph = document.get(CorefChainAnnotation.class);
    if( debug ) System.out.println("coref = " + graph);

    // Analyze the parses with entities
    //List<EntityMention> mentions = stanfordCoref.processParses( trees );

    // For each entity.
    List<EntityMention> mymentions = new ArrayList<EntityMention>();
    for( Integer eid : graph.keySet() ) {
      CorefChain chain = graph.get(eid);
      for( CorefMention mention : chain.getMentionsInTextualOrder() ) {
        EntityMention mymention = new EntityMention(mention.sentNum, mention.mentionSpan, mention.startIndex, mention.endIndex-1, eid);
        mymentions.add(mymention);
      }
    }
    // Put the entities into the doc, and save the mentions that are arguments of verbs.
    if( debug ) System.out.println("MYMENTIONS = " + mymentions);
    mapCorefToTrees(corefdoc, trees, mymentions);
  }

  /**
   * Converts a Stanford pipeline NE tag to our internal type.
   */
  private NERSpan.TYPE nerStringToType(String str) {
    if( str == null ) return null;
    else if( str.equals("ORGANIZATION") ) return NERSpan.TYPE.ORGANIZATION;
    else if( str.equals("PERSON") ) return NERSpan.TYPE.PERSON;
    else if( str.equals("LOCATION") ) return NERSpan.TYPE.LOCATION;
    else return null; //NERSpan.TYPE.NONE;
  }
  
  private void mapCorefToTrees( GigaDoc doc, List<Tree> trees, Collection<EntityMention> entities ) {
    if( entities != null ) {

      // Convert all entity spans from character spans to word-based
      for( EntityMention mention : entities ) {
//        mention.convertCharSpanToIndex(TreeOperator.toRaw(trees[mention.sentenceID()-1]));
        doc.addEntity(mention);
        //  mentions[mention.sentenceID()-1].add(mention);
      }

      // Save the verbs in each parse as events.
      int sid = 0, eid = 0;
      for( Tree tree : trees ) {
        if( tree != null ) {
          // Look for the verbs in the tree
          Vector<Tree> parseVerbs = TreeOperator.verbTreesFromTree(tree);
          for( Tree verb : parseVerbs ) {
            //            System.out.println("  verb: " + verb + " index: " + (TreeOperator.wordIndex(tree,verb)+1));
            doc.addEvent(new WordEvent(eid, verb.firstChild().value(), TreeOperator.wordIndex(tree,verb)+1, sid+1));
            eid++;
          }
          sid++;
        }
      }
    }
  }

  private void continueParsing(String files[], String continueFile) {

    for( String file : files ) {
      if( file.contains(continueFile) ) {

        String parseFile = _outputDir + File.separator + file + ".parse";
        String depsFile = _outputDir + File.separator + file + ".deps";
        String nerFile = _outputDir + File.separator + file + ".ner";
        String corefFile = _outputDir + File.separator + file + ".events";

        GigaDocReader parsed = new GigaDocReader(parseFile);
        int stoppedNum = parsed.largestStoryID(parseFile);
        String stoppedStory = parsed.lastStoryName(parseFile);

        System.out.println("Recovering at " + stoppedStory);

        // If we couldn't find any parses in this file
        if( stoppedNum == 0 ) {
          System.err.println("No parses in the file " + file);
        } else {
          parsed.close();

          // Open the gigaword file
          DocumentHandler giga;
          giga = new GigawordHandler(_dataPath + File.separator + file);
          //	  giga = new GigawordFilteredText(_dataPath + File.separator + file,
          //					    filterList);

          // Find the document at which we last stopped
          giga.nextStory();
          while( giga.currentStory() != null &&
              !giga.currentStory().equals(stoppedStory) ) {
            giga.nextStory();
            System.out.println("checking " + giga.currentStory());
          }
          // Now read the next story
          Vector<String> sentences = giga.nextStory();	  

          GigaDoc doc = null;
          GigaDoc depdoc = null;
          GigaDoc nerdoc = null;
          GigaDoc corefdoc = null;
          try {
            // Create the parse output file.
            doc = new GigaDoc(parseFile, true);
            // Create the dependency output file.
            depdoc = new GigaDoc(depsFile, true);
            nerdoc = new GigaDoc(nerFile, true);
            // Create the coref output file.
            corefdoc = new GigaDoc(corefFile, true);
          } catch( Exception ex ) {
            System.out.println("Skipping to next file...");
            continue;
          }

          // Process the remaining sentences
          int storyID = stoppedNum + 1;
          while( sentences != null && sentences.size() > 0 ) {
            System.out.println(giga.currentDoc() + "/" + giga.numDocs() + " " + giga.currentStory());

            doc.openStory(giga.currentStory(), storyID);
            depdoc.openStory(giga.currentStory(), storyID);
            nerdoc.openStory(giga.currentStory(), storyID);
            corefdoc.openStory(giga.currentStory(), storyID);
            analyzeSentences(giga.currentStory(), giga.currentDoc(), sentences, doc, depdoc, corefdoc, nerdoc);
            doc.closeStory();
            depdoc.closeStory();
            nerdoc.closeStory();
            corefdoc.closeStory();

            sentences = giga.nextStory();
            storyID++;
          }

          doc.closeDoc();
          depdoc.closeDoc();
          nerdoc.closeDoc();
        }
      }
    }
  }

  /**
   * @desc Parse each sentence and save to another file
   */
  public void parseData() {
//    String line = "I am a happy man.";
//    
//    List<Word> allWords = (PTBTokenizer.newPTBTokenizer(new StringReader(line))).tokenize();
//    ArrayList<Word> sentence = new ArrayList<Word>(allWords);
//    System.out.println("allWords: " + allWords);
//    System.out.println("sentence: " + sentence);
//    parser.getOp().testOptions.maxLength = 80;
//    parser.getOp().testOptions.verbose = true;
//    Tree t = parser.parseTree(sentence);
//    System.out.println("treE: " + t);
//
//    
//    String[] options = {"-retainNPTmpSubcategories"};
//    LexicalizedParser lp = LexicalizedParser.loadModel("/home/nchamber/code/resources/englishPCFG.ser.gz");
////    LexicalizedParser lp = new LexicalizedParser("/home/nchamber/code/resources/englishPCFG.ser.gz", options);
//    PTBTokenizer<Word> ptb = PTBTokenizer.newPTBTokenizer(new StringReader(line));
//    List<Word> words = ptb.tokenize();
//    Tree parseTree = lp.parseTree(words);
//    System.out.println("tree: " + parseTree);
//    
//    System.exit(-1);
    
    int numDocs = 0;

    if( _dataPath.length() > 0 && Directory.fileExists(_dataPath) ) {

      String dirPath = null;
      
      // Files array will store list of document files to process.
      String files[] = null;
      File dir = new File(_dataPath);
      if( dir.isDirectory() ) {
        files = Directory.getFilesSorted(_dataPath);
        dirPath = _dataPath;
      } else {
        files = new String[1];;
        files[0] = Directory.filename(_dataPath);
        dirPath = Directory.dirname(_dataPath); 
      }

      // Special handling of a continuation parse.
      if( _continueFile != null ) {
        continueParsing(files, _continueFile);
        return;
      }

      for( String file : files ) {
        System.out.println("file: " + file);
        if( validFilename(file) ) {
          //	  if( file.contains("_2005") ) {

          String parseFile = _outputDir + File.separator + file + ".parse";
          String depsFile = _outputDir + File.separator + file + ".deps";
          String nerFile = _outputDir + File.separator + file + ".ner";
          String corefFile = _outputDir + File.separator + file + ".events";

          Directory.createDirectory(_outputDir);

          if( GigaDoc.fileExists(parseFile) ) {
            System.out.println("File exists: " + parseFile);
          } else {
            System.out.println("file: " + file);

            GigaDoc doc = null;
            GigaDoc depdoc = null;
            GigaDoc nerdoc = null;
            GigaDoc corefdoc = null;

            try {
              // Create the parse output file.
              doc = new GigaDoc(parseFile);
              // Create the dependency output file.
              depdoc = new GigaDoc(depsFile);
              nerdoc = new GigaDoc(nerFile);
              // Create the dependency output file.
              corefdoc = new GigaDoc(corefFile);
            } catch( Exception ex ) {
              System.out.println("Skipping to next file...");
              continue;
            }

            // Open the text file to parse.
            DocumentHandler giga = null;
            String outpath = dirPath + File.separator + file;
            if( _docType == GIGAWORD )
              giga = new GigawordHandler(outpath);
            else if( _docType == ENVIRO )
              giga = new EnviroHandler(outpath);
            else if( _docType == MUC )
              giga = new MUCHandler(outpath);
            else if( _docType == TEXT )
              giga = new TextHandler(outpath);

            // Read the documents in the text file.               else if( _docType == MUC )
            //              giga = new MUCHandler(_dataPath + File.separator + file);
            Vector<String> sentences = giga.nextStory();
            if( debug ) System.out.println("Allparser: got " + sentences);
            int storyID = 0;
            while( sentences != null && sentences.size() > 0 ) {
              //System.out.println("in the while loop for " + file);
              numDocs++;
              System.out.println(numDocs + ": (" + giga.currentDoc() + "/" + giga.numDocs() + ") " + giga.currentStory());
              if( numDocs % 100 == 0 ) Util.reportMemory();

              //		  for( String sentence : sentences ) 
              //		    System.out.println("**" + sentence + "**");

              doc.openStory(giga.currentStory(), storyID);
              depdoc.openStory(giga.currentStory(), storyID);
              nerdoc.openStory(giga.currentStory(), storyID);
              corefdoc.openStory(giga.currentStory(), storyID);
              analyzeSentences(giga.currentStory(), storyID, sentences, doc, depdoc, corefdoc, nerdoc);
              doc.closeStory();
              depdoc.closeStory();
              nerdoc.closeStory();
              corefdoc.closeStory();

              sentences = giga.nextStory();
              storyID++;
            }

            doc.closeDoc();
            depdoc.closeDoc();
            nerdoc.closeDoc();
          }
        }
      }
    }
    else System.err.println("ERROR: bad file path of documents: " + _dataPath);
  }

  /**
   * @param file A filename, not the complete path.
   * @returns True if the given filename matches the requirements of whatever type
   *          of documents we are currently processing.  False otherwise.
   */
  private boolean validFilename(String file) {
    if( !file.startsWith(".") ) {
      if( _docType == GIGAWORD ) {
        if( file.endsWith(".gz") || file.endsWith(".txt") ) return true;
      }
      else if( _docType == ENVIRO ) {
        if( file.endsWith(".txt") ) return true;
      }
      else if( _docType == MUC ) {
        if( file.contains("-muc") ) return true;
      }
      else if( _docType == TEXT ) {
        if( file.contains("txt") ) return true;
      }
    }
	System.out.println("AllParser is skipping " + file + " due to unrecognized file name.");
    return false;
  }


  public static void main(String[] args) {
    AllParser parser = new AllParser(args);
    parser.parseData();
  }
}
