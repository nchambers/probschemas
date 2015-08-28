package nate.probschemas;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import edu.stanford.nlp.util.StringUtils;
import nate.ProcessedData;
import nate.args.VerbArgCounts;
import nate.muc.MUCEntity;
import nate.muc.MUCKeyReader;
import nate.muc.Template;
import nate.util.HandleParameters;
import nate.util.Util;


/**
 * Sort of a wrapper that just reads data files containing parses and
 * coref entities, calling TemplateExtractor to find templates within the
 * documents in these files.
 *
 * -deps, -events, -parsed
 * Paths to the processed files.
 *
 * -muckey
 * Path to the MUC answer key template file.
 *
 * -domainidf
 * -generalidf
 * Paths to the IDF counts for the domain and general corpus.
 *
 * -wordnet
 * Path to WordNet's location.
 *
 * -domainpmi
 * Path to a file of pairs with PMI scores for the domain.
 *
 * -corpuspmi
 * Path to a file of pairs with PMI scores for a larger corpus.
 *
 * -domaincounts
 * Counts of verb-deps within the MUC domain.
 *
 * -corpuscounts
 * Counts of verb-deps over an entire corpus.
 *
 * -domaincorefcounts
 * Counts of verb-deps with coreferring arguments within the MUC domain.
 *
 * -paircounts
 * Path to the token pair counts from CountTokenPairs.
 *
 * -evalatest
 * If present, then we evaluate on the test set based on the training set's counts.
 */
public class TemplateTester {
  ProcessedData _trainDataReader;
  ProcessedData _testDataReader;
  MUCKeyReader _trainAnswerKey;
  MUCKeyReader _testAnswerKey;
  Map<String, List<Integer>> _totalScores;
  Map<String,Integer> _falsePositiveArgs;
  Map<String,Integer> _falseNegativeArgs;
  HandleParameters _params;
  public Set<String> _filterList = null;
  // If true, tests on the test set, not the train set.
  boolean _evaluateOnTest = false;
  

  public TemplateTester(String args[]) {
    _params = new HandleParameters(args);

    System.out.println("parse file: " + _params.get("-parsed"));
    _trainDataReader = new ProcessedData(_params.get("-parsed"), _params.get("-deps"), _params.get("-events"), _params.get("-ner"));
    _testDataReader  = new ProcessedData(_params.get("-testparsed"), _params.get("-testdeps"), _params.get("-testevents"), _params.get("-testner"));

    // e.g. key-dev-0101.muc4
    _trainAnswerKey = new MUCKeyReader(_params.get("-muckey"));
    _testAnswerKey = new MUCKeyReader(_params.get("-testmuckey"));

    if( _params.hasFlag("-evaltest") ) _evaluateOnTest = true;
    System.out.println("evaluateOnTest\t" + _evaluateOnTest);

    // The global evaluation counts we keep track of during the evaluation.
    _totalScores = new HashMap<String, List<Integer>>();
  }

  public TemplateTester() {
    // The global evaluation counts we keep track of during the evaluation.
    _totalScores = new HashMap<String, List<Integer>>();
  }

  /**
   * This updates the false negative count for *all* current evaluations being stored globally.
   * It assumes the given gold templates are from documents that are (say) KIDNAP, but yet
   * were not input to this run of the system, and so we missed them.
   */
  public void recordFalseNegativeArguments(List<Template> goldTemplates) {
    // Get the number of gold entities.
    List<MUCEntity> goldEntities = getGoldEntities(goldTemplates);

    if( _falseNegativeArgs == null ) _falseNegativeArgs = new HashMap<String,Integer>();
    
    // Update every type of evaluation we are recording (as stored in the global counts).
    for( String type : _totalScores.keySet() ) {
      Integer num = _falseNegativeArgs.get(type);
      if( num == null ) _falseNegativeArgs.put(type, goldEntities.size());
      else _falseNegativeArgs.put(type, num + goldEntities.size());
      System.out.println("Put " + goldEntities.size() + " " + type + " false negatives, now at " + _falseNegativeArgs.get(type));
    }
  }
  
  /**
   * The Stanford Parser converts parentheses into -LRB- and -RRB- which we need to convert
   * back if we want to string compare to the MUC gold strings.
   */
  private static String replaceParentheses(String str) {
    if( str.contains("RB-") || str.contains("rb-") ) {
      str = str.replaceAll("-LRB- ", "(");
      str = str.replaceAll(" -RRB-", ")");
      str = str.replaceAll("-lrb- ", "(");
      str = str.replaceAll(" -rrb-", ")");
    } 
    return str;
  }
  
  private static boolean similarStrings(String one, String two) {
    // Equivalent.
    if( one.equalsIgnoreCase(two) )
      return true;
    
    // Plural version.
    if( one.equalsIgnoreCase(two + "s") || one.equalsIgnoreCase(two + "es") )
      return true;

    // One string just has modifiers, but the same root as the other. 
    if( one.endsWith(two) || two.endsWith(one) )
      return true;
    if( one.endsWith(two + "s") || two.endsWith(one + "s") || one.endsWith(two + "es") || two.endsWith(one + "es") )
      return true;
    
    return false;
  }
  
  private static int duplicatesThatWereWrong(Collection<String> guesses, int[] guessesMatched, boolean debug) {
    int duplicates = 0;
    String[] guessStrings = new String[guesses.size()];
    guessStrings = guesses.toArray(guessStrings);

    // Don't double count words that are duplicates if more than 2 of them.
    boolean[] foundDuplicates = new boolean[guesses.size()];
    for( int ii = 0; ii < foundDuplicates.length; ii++ ) 
      foundDuplicates[ii] = false;
    
    for( int ii = 0; ii < guesses.size()-1; ii++ ) {
      if( guessesMatched[ii] == 0 && foundDuplicates[ii] == false) {
        for( int jj = ii+1; jj < guesses.size(); jj++ ) {
          if( guessesMatched[jj] == 0 ) {
            if( similarStrings(guessStrings[ii], guessStrings[jj]) ) {
              foundDuplicates[jj] = true;
              duplicates++;
              if( debug ) System.out.println("SIMILAR STRINGS: " + guessStrings[ii] + "\t" + guessStrings[jj] + "(" + ii + "," + jj + ")");
            }
          }
        }
      }
    }

    return duplicates;
  }
    
  /**
   * Compares golds to guesses, but if the golds are null, we set a global false positives count.
   * Otherwise return the normal
   * @param templateType kidnap, bombing, attack
   * @param evalType Unique string to save these results in.
   * @return
   */
  public void evaluateEntities(String evalType, List<MUCEntity> golds, Collection<String> guesses) {
    for( String guess : guesses ) System.out.println("  guess " + guess);
    if( golds == null || golds.size() == 0 ) {
      if( _falsePositiveArgs == null ) _falsePositiveArgs = new HashMap<String,Integer>();
      Integer num = _falsePositiveArgs.get(evalType);
      if( num == null ) _falsePositiveArgs.put(evalType, guesses.size());
      else _falsePositiveArgs.put(evalType, num + guesses.size());
      System.out.println("Put " + guesses.size() + " " + evalType + " false positives, now at " + _falsePositiveArgs.get(evalType));
    } else {
      recordEvaluation(evalType, evaluateEntities(golds, guesses));
    }    
  }
  
  /**
   * Basic comparison of golds to guesses.
   * @return An array where the first three cells are counts: 
   *              # unique golds that were guessed (correct), # wrong guesses (false pos), # of unique golds missed (false neg)
   *         The next N cells (number of guesses) are 0 if the guess did not match a gold, 1 if it did.
   *         The next M cells (number of golds) are 0 if the gold was matched by a guess, 1 if it was not.
   */
  public static int[] evaluateEntities(List<MUCEntity> golds, Collection<String> guesses) {
    return evaluateEntities(golds, guesses, false);
  }
  public static int[] evaluateEntities(List<MUCEntity> golds, Collection<String> guesses, boolean debug) {
    if( debug ) System.out.println("Evaluating gold: " + golds);
    if( debug ) System.out.println("Against guessed: " + guesses);

    //    if( golds == null || golds.size() == 0 ) {
    //      System.out.println("No Evaluation");
    //      return null;
    //    }
    //
    //    else {
    int numGold = (golds == null) ? 0 : golds.size();

    // This array stores the number of guesses that match each gold entity.
    // We might have two guesses that describe the same entity...we don't
    // want to count that as two correct guesses, so we track these.
    int[] matched = new int[numGold];
    int notmatched = 0;
    int[] guessesMatched = new int[guesses.size()];

    // Check each guess for a matching gold entity.
    if( guesses != null ) {
      int guessi = 0;
      for( String guess : guesses ) {
        int i = 0;
        boolean foundmatch = false;
        if( golds != null ) {
          for( MUCEntity gold : golds ) {
            if( debug ) System.out.println("Checking " + gold + " vs " + guess);
            if( stringMatchToMUCEntity(gold, guess) ) {
              matched[i]++;
              foundmatch = true;
              if( debug ) System.out.println("  Matched! " + gold + " -- " + guess);
            }
            i++;
          }
        }
        if( !foundmatch ) notmatched++;
        else {
          guessesMatched[guessi] = 1;
//          System.out.println("Setting match " + guessi + " now list is " + Arrays.toString(guessesMatched));
        }
        guessi++;
      }
    }

    if( debug ) System.out.println("Matched list: " + Arrays.toString(matched));

    // Count the number correct and the number missed.    
    int correct      = 0;
    int missed       = 0;
    for( int ii = 0; ii < matched.length; ii++ ) {
      if( matched[ii] > 0 ) correct++;
      
      // If it was not an optional entity.
      else if( !golds.get(ii).isOptional() ) {
        missed++;
        if( debug ) System.out.println("   - missed " + golds.get(ii));
      }
    }
    int incorrect = notmatched;
    
    if( debug ) System.out.println("num incorrect = " + incorrect + " array=" + Arrays.toString(guessesMatched));
    int numdups = duplicatesThatWereWrong(guesses, guessesMatched, debug);
    if( debug && numdups > 0 ) System.out.println("got " + numdups + " duplicate wrong, reducing incorrect count.");
    incorrect -= numdups;
    
    float recall = (numGold > 0) ? (100.0f * (float)correct / (float)numGold) : 0.0f;

    if( debug ) {
      System.out.printf("Correct:     %d/%d = %.1f\n", correct, numGold, recall);
      System.out.println("Incorrect:  " + incorrect);
      System.out.println("Missed:     " + missed);
    }

    int[] counts = new int[3 + guessesMatched.length + matched.length];
    counts[0] = correct;
    counts[1] = incorrect;
    counts[2] = missed;
    for( int i = 0; i < guessesMatched.length; i++ )
      counts[3 + i] = guessesMatched[i];
    for( int i = 0; i < matched.length; i++ )
      counts[3 + guessesMatched.length + i] = matched[i];
    return counts;
  }


  public static MUCEntity stringMatchToMUCEntities(List<MUCEntity> golds, String guess) {
    if( golds != null ) {
      for( MUCEntity gold : golds ) {
        if( stringMatchToMUCEntity(gold, guess) ) return gold;
      }
    }
    return null;
  }

  /**
   * @return True if the guess string matches one of the given entity's
   *         string descriptions.  It doesn't have to be an exact match,
   *         we search for the gold's descriptions as a substring of guess.
   */
  public static boolean stringMatchToMUCEntity(MUCEntity gold, String guess) {
    return stringMatchToMUCEntity(gold, guess, true);
  }
  public static boolean stringMatchToMUCEntity(MUCEntity gold, String guess, boolean warnings) {
    guess = replaceParentheses(guess);
    guess = guess.toLowerCase();
    String guessRightmost = (guess.indexOf(' ') > -1 ? guess.substring(guess.lastIndexOf(' ')+1) : guess); 
    
    for( String mention : gold.getMentions() ) {
      mention = mention.toLowerCase();
      boolean ofmatch = false;
//      System.out.println("  -> " + mention + " vs " + guess);

      // Check if guess subsumes mention.
      int index = guess.indexOf(mention);
      if( index > -1 ) {
        //	System.out.println("  --> gold in guess");
        // Don't let "maid" == "aid"
        if( index == 0 || guess.charAt(index-1) == ' ' ) {
          // Don't let "Party of Ohio" == "Ohio"
          if( !guess.contains("of " + mention) )
            return true;
          else {
            if( warnings ) System.out.println("GOOD MATCH GONE BAD?? gold=" + gold + " guess=" + guess);
            ofmatch = true;
          }
        }
      }

      // Check the inverse (mention subsumes guess).
      index = mention.indexOf(guess);
      if( index > -1 ) {
        //	System.out.println("  --> guess in gold");
        if( index == 0 || mention.charAt(index-1) == ' ' ) {
          if( !mention.contains("of " + guess) )
            return true;
          else {
            if( warnings ) System.out.println("GOOD MATCH GONE BAD?? gold=" + gold + " guess=" + guess);
            ofmatch = true;
          }
        }
      }
      
      // Check if they are long strings, with very minimal edits (typographical errors).
      if( guess.length() > 15 && mention.length() > 15 ) {
        if( Math.abs(guess.length() - mention.length()) < 5 ) {
          if( StringUtils.editDistance(guess, mention) < (guess.length() / 6) ) {
            System.out.println("EDIT DISTANCE MATCH! " + guess + " with gold " + mention);
            return true;
          }
        }
      }
      
      // Finally, if the rightmost word matches, we win. This is probably too broad, but it is what everyone does
      // in this field. It does seem to give credit to a lot of things that should be given credit, but it also
      // makes mistakes.
      String mentionRightmost = (mention.indexOf(' ') > -1 ? mention.substring(mention.lastIndexOf(' ')+1) : mention);
      if( !ofmatch && mentionRightmost.equals(guessRightmost) ) {
        System.out.println("RIGHTMOST MATCH: " + guess + " with gold " + mention);
        return true;
      }
    }

    return false;
  }

  public static <E> List<E> mergeLists(List<E> first, List<E> second) {
    List<E> merged = new ArrayList<E>(first);
    for( E obj : second ) {
      if( !merged.contains(obj) )
        merged.add(obj);
    }
    return merged;
  }

  private List<String> fillN(List<String> objects, List<String> filler, int max) {
    List<String> filled = new ArrayList<String>();
    int i = 0;
    for( String obj : objects ) {
      if( i++ == max ) break;
      filled.add(obj);
    }

    int diff = max - filled.size();
    if( diff < 1 ) return filled;

    // Try adding the fillers in order.
    for( String obj : filler ) {
      boolean duplicate = false;
      // Don't keep this object if it equals one of the current ones.
      for( String keep : objects ) {
        if( keep.equals(obj) ) {
          duplicate = true;
          break;
        }
      }
      // Add the object.
      if( !duplicate ) {
        filled.add(obj);
        diff--;
      }
      // stop adding more
      if( diff == -1 ) break;
    }
    return filled;
  }

//  private void recordFalseNegativeArguments(List<MUCTemplate> goldTemplates, String templateType, int maxargs) {
//    List<MUCEntity> goldEntities = null;
//    if( goldTemplates != null ) {
//      goldEntities = new ArrayList<MUCEntity>();
//      for( MUCTemplate template : goldTemplates ) {
//        for( MUCEntity entity : template.getMainEntities() )
//          if( !goldEntities.contains(entity) ) goldEntities.add(entity);
//      }
//    }
//
//    if( _falseNegativeArgs == null ) _falseNegativeArgs = new HashMap<String,Integer>();
//    Integer num = _falseNegativeArgs.get(templateType);
//    if( num == null ) _falseNegativeArgs.put(templateType, goldEntities.size());
//    else _falseNegativeArgs.put(templateType, num + goldEntities.size());
//    System.out.println("Put " + goldEntities.size() + " " + templateType + " false negatives, now at " + _falseNegativeArgs.get(templateType));
//  }
//  
//  /**
//   * Assumes the given frame didn't match any MUC templates for its document.
//   */
//  private void recordFalsePositiveFrameArguments(Frame frame, String templateType, int maxargs) {
//    List<String> guesses = new ArrayList<String>();
//    for( String guess : frame.arguments() )
//      guesses.add(guess);    
//    // Trim to the first N.
//    Util.firstN(guesses, maxargs);
//    
//    if( _falsePositiveArgs == null ) _falsePositiveArgs = new HashMap<String,Integer>();
//    Integer num = _falsePositiveArgs.get(templateType);
//    if( num == null ) _falsePositiveArgs.put(templateType, guesses.size());
//    else _falsePositiveArgs.put(templateType, num + guesses.size());
//    System.out.println("Put " + guesses.size() + " " + templateType + " false positives, now at " + _falsePositiveArgs.get(templateType));
//  }
  
  /**
   * Update the global _totalScores counts by adding these results
   * counts to them.
   * @param type The string key for the global results we are updating.
   * @param results The counts (correct, incorrect, etc.) we update.
   */
  private void recordEvaluation(String type, int[] results) {
    System.out.println("**RECORDING " + type + " " + Arrays.toString(results));
    if( results != null ) {
      List<Integer> scores = _totalScores.get(type);
      if( scores == null ) {
        scores = new ArrayList<Integer>();
        _totalScores.put(type, scores);
      }

      for( int i = 0; i < results.length; i++ ) {
        if( scores.size() < i+1 ) scores.add(results[i]);
        else scores.set(i, scores.get(i) + results[i]);
      }
    }
  }

  public void printEvaluationResults() {
    System.out.println("*****RESULTS*****");

    // Sort the keys.
    String[] keys = new String[_totalScores.size()];
    int i = 0;
    for( String key : _totalScores.keySet() ) keys[i++] = key;
    Arrays.sort(keys);

    // Print the results now, sorted by name.
    for( String key : keys ) {
      
      // Numbers on arg matches only on documents that match KIDNAP correctly.
      List<Integer> scores = _totalScores.get(key);
      System.out.print(key + " :\t");
      if( key.length() < 14 ) System.out.print("\t");
      System.out.print(scores);
      float precision = ((float)scores.get(0) / (float)(scores.get(0)+scores.get(1)));
      float recall = ((float)scores.get(0) / (float)(scores.get(0)+scores.get(2)));
      float f1 = 2.0f * (precision*recall) / (precision+recall);
      System.out.printf("\tprec=%.3f\trecall=%.3f\tf1=%.2f\n", precision, recall, f1);

      // Get false positives and negatives.
      Integer fps = (_falsePositiveArgs == null) ? 0 : _falsePositiveArgs.get(key);
      if( fps == null ) fps = 0;
      Integer fns = (_falseNegativeArgs == null) ? 0 : _falseNegativeArgs.get(key);
      if( fns == null ) fns = 0;
      System.out.println("\tfps=" + fps + "\tfns=" + fns);     

      // Numbers with false positives: all docs that we guessed KIDNAP for.
      precision = ((float)scores.get(0) / (float)(scores.get(0)+scores.get(1)+fps));
      recall = ((float)scores.get(0) / (float)(scores.get(0)+scores.get(2)));
      f1 = 2.0f * (precision*recall) / (precision+recall);
      System.out.printf("\tALL-GUESSED\t[%d, %d, %d] prec=%.3f\trecall=%.3f\tf1=%.2f\n", scores.get(0), scores.get(1)+fps, scores.get(2), precision, recall, f1);

      // Full numbers: include docs where we didn't guess a KIDNAP template at all, and those that we missed.
      precision = ((float)scores.get(0) / (float)(scores.get(0)+scores.get(1)+fps));
      recall = ((float)scores.get(0) / (float)(scores.get(0)+scores.get(2)+fns));
      f1 = 2.0f * (precision*recall) / (precision+recall);
      System.out.printf("\tFULL-DOMAIN\t[%d, %d, %d] prec=%.3f\trecall=%.3f\tf1=%.2f\n", scores.get(0), scores.get(1)+fps, scores.get(2)+fns, precision, recall, f1);
    }
  }

  /**
   * Builds a "bag of entities" from the given templates, taking the union of
   * all entities in the templates.
   * @param templates A list of templates in the same domain.
   * @return The union of all entities in the templates.
   */
  public static List<MUCEntity> getGoldEntities(List<Template> templates) {
    List<MUCEntity> goldEntities = null;
    if( templates != null ) {
      goldEntities = new ArrayList<MUCEntity>();
      for( Template template : templates ) {
        for( MUCEntity entity : template.getMainEntities() )
          if( !goldEntities.contains(entity) ) goldEntities.add(entity);
      }
    }
    else System.out.println("**No templates for the story!");
    return goldEntities;
  }
  
  /**
   * How many cells are marked as both correct?
   */
  private int overlapCorrect(int[] scores, int[] filascores) {
    int overlap = 0;
    if( scores != null && filascores != null ) {
      for( int i = 0; i < scores.length; i++ )
        if( scores[i] == 1 && filascores[i] == 1 )
          overlap++;
    }
    return overlap;
  }

}
