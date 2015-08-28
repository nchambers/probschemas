package nate.probschemas;

import java.io.File;
import java.util.List;

import nate.GigaDoc;
import nate.ProcessedData;
import nate.util.Directory;
import nate.util.HandleParameters;
import nate.util.TreeOperator;
import nate.util.Util;

import edu.stanford.nlp.trees.Tree;

/**
 * Takes gigaword files and creates new files that only contain the main keywords in
 * each document in those files. The output is a parallel version of the gigaword files,
 * same number of documents, but each document is a single sentence with only the keywords.
 *
 * DirectoryTokenizer -output <out-dir> -parse <parse-dir> -dep <deps-dir>
 * 
 * -output
 * Directory to create parsed and dependency files.
 *
 */
public class KeywordTokenizer {
  public int MAX_SENTENCE_LENGTH = 400;
  String _outputDir = ".";
  String _parseDir;
  String _depDir;
  KeywordDetector keyDetector;

  public KeywordTokenizer(String[] args) {
    handleParameters(args);
    keyDetector = new KeywordDetector(); 
  }

  private void handleParameters(String[] args) {
    HandleParameters params = new HandleParameters(args);

    if( !params.hasFlag("-output") || !params.hasFlag("-parse") || !params.hasFlag("-dep") ) {
      System.out.println("KeywordTokenizer -output <out-dir> -parse <parse-dir> -dep <deps-dir>");
      System.exit(-1);
    }
    
    if( params.hasFlag("-output") )
      _outputDir = params.get("-output");
    if( params.hasFlag("-parse") )
      _parseDir = params.get("-parse");
    if( params.hasFlag("-dep") )
      _depDir = params.get("-dep");
  }

  public void tokenizeData() {
    if( _parseDir.length() > 0 && Directory.isDirectory(_parseDir) ) {

      // Loop over the files.
      for( String parseFile : Directory.getFilesSorted(_parseDir) ) {
        if( validFilename(parseFile) ) {
          //	  if( file.contains("_2005") ) {
          
          String depFile = Directory.nearestFile(parseFile, _depDir);
          String tokenFile = _outputDir + File.separator + parseFile + ".tokenized";

          if( GigaDoc.fileExists(tokenFile) ) {
            System.out.println("File exists: " + tokenFile);
          } else {
            System.out.println("file: " + parseFile);

            GigaDoc doc = null;

            // Create the tokenized output file.
            try {
              doc = new GigaDoc(tokenFile);
            } catch( Exception ex ) {
              System.out.println("Skipping to next file...");
              continue;
            }

            String parsesPath = _parseDir + File.separator + parseFile;
            String depPath = _depDir + File.separator + depFile;
            System.out.println(parsesPath + "\t" + depPath);
            ProcessedData data = new ProcessedData(parsesPath, depPath, null, null);
            int sid = 0;
            
            data.nextStory();
            while( data.getParseStrings() != null ) {
 
              List<Tree> trees = TreeOperator.stringsToTrees(data.getParseStrings());
              List<String> keywords = keyDetector.getKeywords(trees, data.getDependencies());

              // Stories must be more than just a brief snippet summary.
              // (short is good, but too short leaves out key entities)
              if( trees.size() > 2 ) {
                doc.openStory(data.currentStory(), sid++);
                if( keywords != null ) doc.addParse(Util.collectionToString(keywords, Integer.MAX_VALUE));
                doc.closeStory();
              }

              // Advance to next story.
              data.nextStory();
            }
            
            doc.closeDoc();
          }
        }
      }
    }
    else System.err.println("Path is not a directory: " + _parseDir);
  }

  /**
   * @param file A filename, not the complete path.
   * @returns True if the given filename matches the requirements of whatever type
   *          of documents we are currently processing.  False otherwise.
   */
  private boolean validFilename(String file) {
    if( !file.startsWith(".") ) {
      if( file.endsWith(".gz") || file.endsWith(".txt") ) 
        return true;
    }
    return false;
  }


  public static void main(String[] args) {
    if( args.length > 0 ) {
      KeywordTokenizer parser = new KeywordTokenizer(args);
      parser.tokenizeData();
    }
  }
}
