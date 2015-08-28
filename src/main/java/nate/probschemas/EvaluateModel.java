package nate.probschemas;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import nate.probschemas.EvaluateTemplates;
import nate.muc.MUCEntity;
import nate.muc.KeyReader;
import nate.muc.MUCTemplate;
import nate.muc.Template;
import nate.util.Util;


public class EvaluateModel {
  KeyReader _answerKey;
  
  List<String> _docNames;
  List<List<TextEntity>> _docGuesses;
  int _numLearnedSlots;
  public boolean _debugOn = false;

  // If true, evaluation only looks at documents that have a template.
  // Documents with no templates are removed (don't count any false positives in them).
  public boolean _evaluateOnTemplateDocsOnly = false; 

  
  /**
   * Assumes we will evaluate every template you give, no filtering later.
   */
  public EvaluateModel(int numLearnedSlots, KeyReader answerKey) {
    _answerKey = answerKey;
    _numLearnedSlots = numLearnedSlots;
  }
  
  public void setGuesses(List<String> docNames, List<List<TextEntity>> docGuesses) {
    _docNames = docNames;
    _docGuesses = docGuesses;
  }

  /**
   * Evaluates all gold slots that are currently loaded into this class, regardless
   * of if the gold templates are different event types.
   */
  public void evaluateSlotsIgnoringEventTypes() {
    // How many slots in our current domain?
    List<Integer> slotids = new ArrayList<Integer>();
    int numDomainSlots = _answerKey.numSlots();
 
    // How many slots in our learned model?
    int[] modelids = new int[_numLearnedSlots];
    for( int i = 0; i < _numLearnedSlots; i++ )
      modelids[i] = -1;

    // Possible alignments from learned slots to template slots.
    List<int[]> perms = EvaluateTemplates.permutations(slotids, modelids);

    // Test each permutation to find the best.
    int i = 0;
    double bestf1 = -1.0;
    int[] bestperm = null;
    for( int[] perm : perms ) {
      if( howManyOn(perm) == numDomainSlots ) {
        System.out.println("Trying perm=" + Arrays.toString(perm));
        double[] avgPRF1 = evaluateSlotsIgnoringEventTypes(perm, true);
        if( avgPRF1[2] > bestf1 ) {
            bestf1 = avgPRF1[2];
            bestperm = perm;
        }
      }
      else System.out.println("Skipping perm=" + Arrays.toString(perm));
      i++;
      //      if( i == 5 ) break;
    }

    // Run again with debugging turned on.
    System.out.println("Best permutation is " + Arrays.toString(bestperm) + " at avgF1=" + bestf1);
    double[] avgPRF1 = evaluateSlotsIgnoringEventTypes(bestperm, true);
  }
  
  
  private int howManyOn(int[] perm) {
    int count = 0;
    for( int ii = 0; ii < perm.length; ii++ )
      if( perm[ii] > -1 ) count++;
    return count;
  }
  
  /**
   * Determines if the given role is part of a template that already has a role mapped to 
   * a gold slot. If yes, returns true. If not (it is the first role in its template), returns false.
   * @param perm The role->slot mapping array. 
   * @param rolei The index of the role we care about.
   * @param numTemplates The total number of templates.
   * @return True if the given role index is part of a template that already has a role turned on in the perm array.
   */
  private boolean roleTemplateAlreadyInUse(int[] perm, int rolei, int numTemplates) {
  	int numRolesPerTemplate = perm.length / numTemplates;
  	int targetTemplate = rolei / numRolesPerTemplate;
//  	System.out.println(targetTemplate + " " + numRolesPerTemplate);
  	for( int ii = 0; ii < numRolesPerTemplate; ii++ )
			if( perm[targetTemplate*numRolesPerTemplate+ii] > -1 )
				return true;
  	return false;
  }
  
  /**
   * Determines if a template has a role that is mapped to the given slot.
   * The function takes a role index, finds its template, then looks for a role in it that is mapped to sloti.
   * @param perm The current role->slot mapping.
   * @param rolei The role index in the desired template.
   * @param sloti The slot index we check for.
   * @param numTemplates The number of templates learned.
   * @return True if the role's template already has a role mapping to sloti.
   */
  private boolean roleTemplateMapsToSlot(int[] perm, int rolei, int sloti, int numTemplates) {
    int numRolesPerTemplate = perm.length / numTemplates;
    int targetTemplate = rolei / numRolesPerTemplate;
    for( int ii = 0; ii < numRolesPerTemplate; ii++ )
      if( perm[targetTemplate*numRolesPerTemplate+ii] == sloti )
        return true;
    return false;
  }  
  
  /**
   * Counts how many templates have at least one role mapped to a slot in the given perm array.
   * @param perm The role->slot mapping array. Buckets with -1 mean the role has not been mapped. 
   * @param numTemplates The total number of templates.
   * @return The number of templates that have a mapped role.
   */
  private int numTemplatesInUse(int[] perm, int numTemplates) {
  	int numon = 0;
  	int numRolesPerTemplate = perm.length / numTemplates;
  	for( int xx = 0; xx < numTemplates; xx++ ) {
  		for( int ii = 0; ii < numRolesPerTemplate; ii++ ) {
  			if( perm[xx*numRolesPerTemplate+ii] > -1 ) {
  				numon++;
  				break;
  			}  				
  		}
  	}
  	return numon;
  }
  
  /**
   * Evaluates all gold slots that are currently loaded into this class, regardless
   * of if the gold templates are different event types.
   * @return Three values: precision, recall, F1.
   */
  public double[] evaluateSlotsGreedy(int maxRolesPerSlot) {
    System.out.println("debugon = " + _debugOn);
  
    // Find the best scoring permutation.
    int[] bestperm = greedyMultipleSlots(_answerKey.numSlots(), _numLearnedSlots, maxRolesPerSlot);
  
    // Run AGAIN purely for debugging output.
    System.out.println("** Best perm: " + Arrays.toString(bestperm) + " **");
    double[] avgPRF1 = evaluateSlotsIgnoringEventTypes(bestperm, true);
    return avgPRF1;
  }

  /**
   * Evaluates all gold slots that are currently loaded into this class, regardless
   * of if the gold templates are different event types.
   * @param maxTemplatesToMap The number of templates allowed to use to map learned roles to slots.
   * @return Three values: precision, recall, F1.
   */
  public double[] evaluateSlotsAsTemplatesGreedy(int maxTemplatesToMap, int numTemplates) {
    System.out.println("debugon = " + _debugOn);
  
    // Find the best scoring permutation.
    int[] bestperm = greedyMultipleTemplates(_answerKey.numSlots(), _numLearnedSlots, numTemplates, maxTemplatesToMap);
  
    // Run AGAIN purely for debugging output.
    System.out.println("** Best perm: " + Arrays.toString(bestperm) + " **");
    double[] avgPRF1 = evaluateSlotsIgnoringEventTypes(bestperm, true);
    return avgPRF1;
  }

  public double[] evaluateSlotsBestSingleRoleEachSlot() {
    System.out.println("debugon = " + _debugOn);

    // Find the best scoring permutation.
    int[] bestperm = permWithSingleRolePerTemplateSlot(_numLearnedSlots);
  
    // Run AGAIN purely for debugging output.
    System.out.println("** Best perm: " + Arrays.toString(bestperm) + " **");
    double[] avgPRF1 = evaluateSlotsIgnoringEventTypes(bestperm, true);
    return avgPRF1;
  }

  public double[] evaluateSlotsBestSchemaForEachMUCType(int numTemplates) {
    System.out.println("debugon = " + _debugOn);

    // Find the best scoring permutation.
    int[] bestperm = permWithTemplateMapping(_numLearnedSlots, numTemplates);
  
    // Run AGAIN purely for debugging output.
    System.out.println("** Best perm: " + Arrays.toString(bestperm) + " **");
    double[] avgPRF1 = evaluateSlotsIgnoringEventTypes(bestperm, true);
    return avgPRF1;
  }
  
  /**
   * Maps as many learned roles as you want (maxRolesPerSlot) to any single gold slot.
   * Keeps mapping learned roles to slots until the F1 stops improving or the max is reached.
   * @param numSlots The number of gold MUC template slots.
   * @param numRoles The number of learned slots.
   * @param maxRolesPerSlot The number of learned roles that are allowed to map to any given slot.
   * @return An array with length numRoles, and each int is -1 if it is not mapped. If it is mapped,
   *         then the number is the index from 0 to numSlots indicating the MUC slot it mapped to. 
   */
  private int[] greedyMultipleSlots(int numSlots, int numRoles, int maxRolesPerSlot) {
    // Score each slot/role mapping, and sort by F1.
    Map<String,Double> f1scores = new HashMap<String,Double>();
    Map<String,double[]> fullscores = new HashMap<String,double[]>();
    for( int sloti = 0; sloti < numSlots; sloti++ ) {
      for( int rolei = 0; rolei < numRoles; rolei++ ) {
        double[] prf1 = evaluateSlot(sloti, rolei, _debugOn);
        // Don't bother adding F1 scores below .003 (arbitrary small number)
        if( !Double.isNaN(prf1[2]) && prf1[2] > .003 ) {
          f1scores.put(sloti + " " + rolei, prf1[2]);
          fullscores.put(sloti + " " + rolei, prf1);
          System.out.println("** GREEDY Added pair: " + sloti + " " + rolei + " " + prf1[2]);
        }
        else System.out.println("** GREEDY Skipping NaN pair: " + sloti + " " + rolei);
      }
    }
    List<String> sorted = Util.sortKeysByValues(f1scores);
    
    // Initialize role->slot mapping.
    int[] perm = new int[numRoles];
    for( int x = 0; x < numRoles; x++ ) perm[x] = -1;
    
    // Greedy match roles to slots (multiple roles can go to the same slot).
    double[] currentPRF1 = null;
    for( int index = 0; index < sorted.size(); index++ ) {
      String pair = sorted.get(index);
      Integer sloti = Integer.valueOf(pair.split(" ")[0]);
      Integer rolei = Integer.valueOf(pair.split(" ")[1]);
      
      // Check that we haven't mapped too many roles to this slot already.
      if( numTimesSlotIsMapped(perm, sloti) < maxRolesPerSlot ) {
        System.out.println("*GREEDY NOW TRIES (" + index + " of " + sorted.size() + "): " + pair + "\t" + Arrays.toString(fullscores.get(pair)));

        // If never mapped this role, map it!
        if( !permMapped(perm, rolei) ) {
          perm[rolei] = sloti;
          double[] newPRF1 = evaluateSlotsIgnoringEventTypes(perm, false);
          System.out.println("*GREEDY TRIED " + Arrays.toString(perm));
          System.out.println("*GREEDY GOT NEW newPRF1 = " + Arrays.toString(newPRF1));
          if( currentPRF1 != null && newPRF1[2] <= currentPRF1[2] )
            perm[rolei] = -1;
          else
            currentPRF1 = newPRF1;
        }
        // If role already mapped, try this different mapping and see if it is better.
        else {
          int oldsloti = perm[rolei];
          perm[rolei] = sloti;
          double[] newPRF1 = evaluateSlotsIgnoringEventTypes(perm, false);
          System.out.println("*GREEDY TRIED " + Arrays.toString(perm));
          System.out.println("*GREEDY GOT REPLACE newPRF1 = " + Arrays.toString(newPRF1));
          if( newPRF1[2] < currentPRF1[2] )
            perm[rolei] = oldsloti;
          else
            currentPRF1 = newPRF1;
        }
      } else System.out.println("*GREEDY OVER-MAPPED SLOT " + sloti + ": SKIPPING pair=" + pair);
    }
    
    // Run again with debugging turned on.
    System.out.println("Best permutation is " + Arrays.toString(perm) + " at avgF1=" + (currentPRF1 == null ? null : currentPRF1[2]));
    return perm;
  }
  
  /**
   * 
   * @param numSlots
   * @param numRoles
   * @param numTemplates The total number of templates that were learned.
   * @param maxTemplatesToMap The number of templates allowed to use to map learned roles to slots.
   * @return
   */
  private int[] greedyMultipleTemplates(int numSlots, int numRoles, int numTemplates, int maxTemplatesToMap) {
    // Score each slot/role mapping, and sort by F1.
    Map<String,Double> f1scores = new HashMap<String,Double>();
    Map<String,double[]> fullscores = new HashMap<String,double[]>();
    for( int sloti = 0; sloti < numSlots; sloti++ ) {
      for( int rolei = 0; rolei < numRoles; rolei++ ) {
        double[] prf1 = evaluateSlot(sloti, rolei, _debugOn);
        // Don't bother adding F1 scores below .003 (arbitrary small number)
        if( !Double.isNaN(prf1[2]) && prf1[2] > .003 ) {
          f1scores.put(sloti + " " + rolei, prf1[2]);
          fullscores.put(sloti + " " + rolei, prf1);
          System.out.println("** GREEDY Added pair: " + sloti + " " + rolei + " " + prf1[2]);
        }
        else System.out.println("** GREEDY Skipping NaN pair: " + sloti + " " + rolei);
      }
    }
    List<String> sorted = Util.sortKeysByValues(f1scores);
    
    // Initialize role->slot mapping.
    int[] perm = new int[numRoles];
    for( int x = 0; x < numRoles; x++ ) perm[x] = -1;
    
    // Greedy match roles to slots (multiple roles can go to the same slot).
    double[] currentPRF1 = null;
    for( int index = 0; index < sorted.size(); index++ ) {
      String pair = sorted.get(index);
      Integer sloti = Integer.valueOf(pair.split(" ")[0]);
      Integer rolei = Integer.valueOf(pair.split(" ")[1]);
      
      // Check that we haven't mapped too many roles to this slot already.
//      if( numTimesSlotIsMapped(perm, sloti) < maxRolesPerSlot ) {
      if( (roleTemplateAlreadyInUse(perm, rolei, numTemplates) || numTemplatesInUse(perm, numTemplates) < maxTemplatesToMap) && 
          !roleTemplateMapsToSlot(perm, rolei, sloti, numTemplates) ) {
        System.out.println("*GREEDY NOW TRIES (" + index + " of " + sorted.size() + "): " + pair + "\t" + Arrays.toString(fullscores.get(pair)));
        
        // If never mapped this role, map it!
        if( !permMapped(perm, rolei) ) {
          perm[rolei] = sloti;
          double[] newPRF1 = evaluateSlotsIgnoringEventTypes(perm, false);
          System.out.println("*GREEDY TRIED " + Arrays.toString(perm));
          System.out.println("*GREEDY GOT NEW newPRF1 = " + Arrays.toString(newPRF1));
          if( currentPRF1 != null && newPRF1[2] <= currentPRF1[2] )
            perm[rolei] = -1;
          else
            currentPRF1 = newPRF1;
        }
        // If role already mapped, try this different mapping and see if it is better.
        else {
          int oldsloti = perm[rolei];
          perm[rolei] = sloti;
          double[] newPRF1 = evaluateSlotsIgnoringEventTypes(perm, false);
          System.out.println("*GREEDY TRIED " + Arrays.toString(perm));
          System.out.println("*GREEDY GOT REPLACE newPRF1 = " + Arrays.toString(newPRF1));
          if( newPRF1[2] < currentPRF1[2] )
            perm[rolei] = oldsloti;
          else
            currentPRF1 = newPRF1;
        }
      } else System.out.println("*GREEDY OVER-MAPPED SLOT " + sloti + ": SKIPPING pair=" + pair);
    }
    
    // Run again with debugging turned on.
    System.out.println("Best permutation is " + Arrays.toString(perm) + " at avgF1=" + (currentPRF1 == null ? null : currentPRF1[2]));
    return perm;
  }
  
  /**
   * Quick and dirty. Of course, best results not guaranteed.
   * (1) Score each slot/role mapping in isolation. 
   * (2) Add each slot/role mapping to the final permutation in order of the highest F1 scores, non-repeating of slots.
   * @param numSlots The number of slots in the gold template structure.
   * @param numRoles The number of learned slots.
   */
  private int[] greedySlotAlignment(int numSlots, int numRoles) {
    Map<String,Double> scores = new HashMap<String,Double>();
    
    for( int sloti = 0; sloti < numSlots; sloti++ ) {
      for( int rolei = 0; rolei < numRoles; rolei++ ) {
        double[] prf1 = evaluateSlot(sloti, rolei, true);
        if( !Double.isNaN(prf1[2]) ) {
          scores.put(sloti + " " + rolei, prf1[2]);
          System.out.println("** Added pair: " + sloti + " " + rolei + " " + prf1[2]);
        }
        else System.out.println("** Skipping NaN pair: " + sloti + " " + rolei);
      }
    }
    
    Set<Integer> slotsMapped = new HashSet<Integer>();
    Set<Integer> rolesMapped = new HashSet<Integer>();
    int[] perm = new int[numRoles];
    for( int xx = 0; xx < numRoles; xx++ ) perm[xx] = -1;
    
    List<String> sorted = Util.sortKeysByValues(scores);
    for( String pair : sorted ) {
      System.out.println("** " + pair + "\t" + scores.get(pair));
      Integer sloti = Integer.valueOf(pair.split(" ")[0]);
      Integer rolei = Integer.valueOf(pair.split(" ")[1]);
      if( !slotsMapped.contains(sloti) && !rolesMapped.contains(rolei) ) {
        slotsMapped.add(sloti);
        rolesMapped.add(rolei);
        perm[rolei] = sloti;
      }
    }
    return perm;
  }

  /**
     * Incrementally builds role/slot permutations, adding the best scoring ones first.
     * Then it clones that permutation for different mappings, generating hundreds of mappings.
     * The best scoring one is returned.
     * @param numSlots
     * @param numRoles
     * @return
     */
    private int[] greedyBestSoFarAlignment(int numSlots, int numRoles) {
      // Score each slot/role mapping, and sort by F1.
      Map<String,Double> scores = new HashMap<String,Double>();
      for( int sloti = 0; sloti < numSlots; sloti++ ) {
        for( int rolei = 0; rolei < numRoles; rolei++ ) {
          double[] prf1 = evaluateSlot(sloti, rolei, _debugOn);
          // Don't bother adding F1 scores below .003 (arbitrary small number)
          if( !Double.isNaN(prf1[2]) && prf1[2] > .003 ) {
            scores.put(sloti + " " + rolei, prf1[2]);
            System.out.println("** Added pair: " + sloti + " " + rolei + " " + prf1[2]);
          }
          else System.out.println("** Skipping NaN pair: " + sloti + " " + rolei);
        }
      }
      List<String> sorted = Util.sortKeysByValues(scores);
      
      // Run through our permutations.
      List<String> seenPerms = new ArrayList<String>();
      List<int[]> knownPerms = new ArrayList<int[]>();
      List<Double> permScores = new ArrayList<Double>();
  
      // Initialize any empty permutation.
      int[] firstPerm = new int[numRoles];
      for( int xx =0 ; xx < firstPerm.length; xx++ ) firstPerm[xx] = -1;
      knownPerms.add(firstPerm);
      seenPerms.add(Arrays.toString(firstPerm));
      permScores.add(0.0);
      
      // Add each scored mapping incrementally.
      for( int index = 0; index < sorted.size(); index++ ) {
        String pair = sorted.get(index);
        Integer sloti = Integer.valueOf(pair.split(" ")[0]);
        Integer rolei = Integer.valueOf(pair.split(" ")[1]);
        System.out.println("** " + pair + "\t" + scores.get(pair));
  
        int numPerms = knownPerms.size();        
        for( int permi = 0; permi < numPerms; permi++ ) {
          int[] perm = knownPerms.get(permi);
  //        System.out.println("permi=" + permi + "\t" + Arrays.toString(perm));
  //        System.out.println("Checking permi=" + permi + " perm=" + Arrays.toString(perm));
          if( !permContains(perm, sloti) && !permMapped(perm, rolei) ) {
            int[] newperm = Arrays.copyOf(perm, perm.length);
            newperm[rolei] = sloti;
            if( !seenPerms.contains(Arrays.toString(newperm)) ) {
              knownPerms.set(permi, newperm);
              permScores.set(permi, scoreOfPerm(newperm, scores));
              seenPerms.add(Arrays.toString(newperm));
              //          System.out.println("Changed perm=" + Arrays.toString(perm));
            }
          }
          else if( !permContains(perm, sloti) && permMapped(perm, rolei) ) {
            int[] newperm = Arrays.copyOf(perm, perm.length);
            newperm[rolei] = sloti;
            if( !seenPerms.contains(Arrays.toString(newperm)) ) {
              knownPerms.add(newperm);
              seenPerms.add(Arrays.toString(newperm));
              //          System.out.println("permi=" + permi + " pair=" + pair);
              permScores.add(scoreOfPerm(newperm, scores));
              //          System.out.println("Added perm=" + Arrays.toString(newperm));
            }
          }
          else if( permContains(perm, sloti) && !permMapped(perm, rolei) ) {
            int[] newperm = Arrays.copyOf(perm, perm.length);
            newperm[rolei] = sloti;
            int oldrole = permContainsAtIndex(perm, sloti);
            newperm[oldrole] = -1;
            if( !seenPerms.contains(Arrays.toString(newperm)) ) {
              knownPerms.add(newperm);
              seenPerms.add(Arrays.toString(newperm));
              permScores.add(scoreOfPerm(newperm, scores));
              //          System.out.println("Subbed perm=" + Arrays.toString(newperm));
            }
          }
        }
      }
      
      // Find the best score.
      double best = -1.0;
      int[] bestperm = null;
      for( int xx = 0; xx < permScores.size(); xx++ ) {
  //      System.out.println("permi=" + xx + "\tscore=" + permScores.get(xx) + "\t" + Arrays.toString(knownPerms.get(xx)));
        if( permScores.get(xx) > best ) {
          best = permScores.get(xx);
          bestperm = knownPerms.get(xx);
        }
      }
      
      // Run again with debugging turned on.
      System.out.println("Best permutation is " + Arrays.toString(bestperm) + " at avgF1=" + best);
  //    double[] avgPRF1 = evaluateSlotsIgnoringEventTypes(bestperm, true);
      
      return bestperm;
    }

  /**
   * NOTE: this does not correctly calculate F1. It simply sums up F1 scores for each slot.
   * This is obviously wrong as slots occur with different frequency...
   * @param perm A slot/role mapping for all learned roles.
   * @param scores A mapping from pairs "slot role" to the F1 of that mapping.
   * @return The sum of F1 scores of the given perm.
   */
  private double scoreOfPerm(int[] perm, Map<String,Double> scores) {
    double score = 0.0;
    for( int rolei = 0; rolei < perm.length; rolei++ ) {
      if( perm[rolei] > -1 ) {
        String pair = perm[rolei] + " " + rolei;
        Double pscore = scores.get(pair); 
        if( pscore != null )
          score += pscore; 
      }
    }
    System.out.println("--scored perm=" + Arrays.toString(perm) + "\t" + score);
    return score;
  }
  
  private int permContainsAtIndex(int[] perm, int sloti) {
    for( int ii = 0; ii < perm.length; ii++ )
      if( perm[ii] == sloti ) return ii;
    return -1;
  }
  
  private boolean permContains(int[] perm, int sloti) {
    for( int val : perm )
      if( val == sloti ) return true;
    return false;
  }

  private boolean permMapped(int[] perm, int rolei) {
    if( perm[rolei] < 0 ) return false;
    else return true;
  }

  private int numTimesSlotIsMapped(int[] perm, int sloti) {
    int count = 0;
    for( int val : perm )
      if( val == sloti ) 
        count++;
    return count;
  }

  /**
   * Find the best learned role for each SPECIFIC template slot (kidnap-perp, not just general perp).
   * @param numLearnedRoles
   * @return
   */
  private int[] permWithSingleRolePerTemplateSlot(int numLearnedRoles) {
//	final String[] types = { "KIDNAP", "BOMBING", "ATTACK", "FORCED WORK STOPPAGE", "ROBBERY", "ARSON" };
  	final String[] types = { "KIDNAP", "BOMBING", "ATTACK", "ARSON" };

  	Map<String,Integer> slotToRole = new HashMap<String,Integer>();
  	Map<String,double[]> slotScores = new HashMap<String,double[]>();
  	
  	int[] perm = new int[numLearnedRoles];
  	for( int xx = 0; xx < numLearnedRoles; xx++ ) perm[xx] = -1;
  	
  	for( int rolei = 0; rolei < numLearnedRoles; rolei++ ) {
  		for( String type : types ) {
  			for( int sloti = 0; sloti < 4; sloti++ ) {
  				double[] prf1 = evaluateSlot(sloti, rolei, type, false);
  				double[] currentBest = slotScores.get(type+sloti);
  				System.out.println("prf1=" + Arrays.toString(prf1) + " prev=" + Arrays.toString(currentBest));
  				if( currentBest == null || Double.isNaN(currentBest[2]) || currentBest[2] < prf1[2] ) {
  					slotScores.put(type+sloti, prf1);
  					slotToRole.put(type+sloti, rolei);
  				}
  			}
  		}
  	}
  	
  	// Now pull them out.
  	System.out.println("*** BEST MUC SLOT TO LEARNED ROLE MAPPING ***");
		for( String type : types ) {
			for( int sloti = 0; sloti < 4; sloti++ ) {
				int bestrolei = slotToRole.get(type+sloti);
				double[] prf1 = slotScores.get(type+sloti);
				System.out.println(type + " " + sloti + " -> role " + bestrolei + " f1=" + Arrays.toString(prf1));
				if( prf1[2] > .01 )
					perm[bestrolei] = sloti;
			}
		}
		System.out.println("final mapping perm=" + Arrays.toString(perm));
  	return perm;
  }
  
  private int[] bestLearnedSchemaForMUCTemplate(int numLearnedRoles, int numTemplates, String type) {
  	int numRolesPerTemplate = numLearnedRoles / numTemplates;

  	double currentBestTemplate = -1.0;
  	int bestTemplate = -1;
  	int[] bestRoles = new int[4];
  	for( int xx = 0; xx < 4; xx++ ) bestRoles[xx] = -1;

  	for( int templatei = 0; templatei < numTemplates; templatei++ ) {
  	  Map<String,Integer> slotToRole = new HashMap<String,Integer>();
  	  Map<String,double[]> slotScores = new HashMap<String,double[]>();
  	  Map<Integer,Double> mappedRolesToBestScore = new HashMap<Integer,Double>();

  	  // GREEDY MAPPING. Add best first.
  	  // Score each slot/role mapping, and then sort all possible mappings by F1.
  	  Map<String,Double> f1scores = new HashMap<String,Double>();
  	  Map<String,double[]> fullscores = new HashMap<String,double[]>();
  	  for( int sloti = 0; sloti < 4; sloti++ ) {
  	    for( int roleOffset = 0; roleOffset < numRolesPerTemplate; roleOffset++ ) {
  	      int rolei = templatei*numRolesPerTemplate + roleOffset;
  	      double[] prf1 = evaluateSlot(sloti, rolei, type, false);
  	      // Don't bother adding F1 scores below .003 (arbitrary small number)
  	      if( !Double.isNaN(prf1[2]) && prf1[2] > .003 ) {
  	        f1scores.put(sloti + " " + rolei, prf1[2]);
  	        fullscores.put(sloti + " " + rolei, prf1);
  	      }
  	    }
  	  }
  	  List<String> sorted = Util.sortKeysByValues(f1scores);

  	  // Now add the top slot/role mappings one by one.
  	  Set<Integer> mappedRoles = new HashSet<Integer>();
  	  for( String slotrole : sorted ) {
  	    Integer sloti = Integer.valueOf(slotrole.split(" ")[0]);
  	    Integer rolei = Integer.valueOf(slotrole.split(" ")[1]);
  	    System.out.println(type + " next: " + sloti + " " + rolei + " = " + Arrays.toString(fullscores.get(slotrole)));
  	    System.out.println("(" + type + ") prf1=" + Arrays.toString(fullscores.get(slotrole)));
  	    Double f1 = f1scores.get(slotrole);
  	    if( !mappedRoles.contains(rolei) ) {
  	      if( f1 != null && !f1.isNaN() && (!slotToRole.containsKey(type+sloti) || slotScores.get(type+sloti)[2] < f1) ) {
  	        slotToRole.put(type+sloti, rolei);
  	        slotScores.put(type+sloti, fullscores.get(slotrole));
  	        mappedRoles.add(rolei);
  	        System.out.println("*Saving new best role " + rolei);
  	      }
  	    }
  	  }

      /*
    	// Try each learned role within just this template.
    	// Keep best role per slot.
  		for( int roleOffset = 0; roleOffset < numRolesPerTemplate; roleOffset++ ) {
  			int rolei = templatei*numRolesPerTemplate + roleOffset;
  			for( int sloti = 0; sloti < 4; sloti++ ) {
  				double[] prf1 = evaluateSlot(sloti, rolei, type, false);
  				double[] currentBest = slotScores.get(type+sloti);
  				System.out.println("(" + type + ") prf1=" + Arrays.toString(prf1) + " prev=" + Arrays.toString(currentBest));
  				if( prf1[2] > 0.0 ) {
  					if( currentBest == null || Double.isNaN(currentBest[2]) || currentBest[2] < prf1[2] ) {
  						// Make sure this role isn't already mapped at a higher score to a different MUC slot.
  						if( !mappedRolesToBestScore.containsKey(rolei) || mappedRolesToBestScore.get(rolei) < prf1[2] ) {
  							System.out.println("*Saving new best role " + rolei);
  							slotScores.put(type+sloti, prf1);
  							slotToRole.put(type+sloti, rolei);
  							mappedRolesToBestScore.put(rolei, prf1[2]);							
  						}
  					}
  				}
  			} 
  		}
       */
      
  	  // Sum F1 scores to get a total template score (hacky macro-average, not always correct but close enough?)
  	  double templateScore = 0.0;
  	  for( int sloti = 0; sloti < 4; sloti++ ) {
  	    double[] prf1 = slotScores.get(type+sloti);
  	    if( prf1 != null && prf1[2] > 0.0 )
  	      templateScore += prf1[2];
  	  }
  	  System.out.println("TEMPLATE SCORE " + type + " templatei=" + templatei + "\t" + templateScore);
  	  // Is it the best?
  	  if( currentBestTemplate < templateScore ) {
  	    currentBestTemplate = templateScore;
  	    bestTemplate = templatei;
  	    for( int sloti = 0; sloti < 4; sloti++ ) {
  	      Integer rolei = slotToRole.get(type+sloti);
  	      if( rolei != null && rolei > -1 ) bestRoles[sloti] = rolei;
  	      else bestRoles[sloti] = -1;
  	    }
  	  }
  	}
  	System.out.println("BEST MUC ROLES " + type + " (template " + bestTemplate + "): " + Arrays.toString(bestRoles));
  	return bestRoles;
  }
  
  private int[] permWithTemplateMapping(int numLearnedRoles, int numTemplates) {
//	final String[] types = { "KIDNAP", "BOMBING", "ATTACK", "FORCED WORK STOPPAGE", "ROBBERY", "ARSON" };
  	final String[] types = { "KIDNAP", "BOMBING", "ATTACK", "ARSON" };
  	
  	int[] perm = new int[numLearnedRoles];
  	for( int xx = 0; xx < numLearnedRoles; xx++ ) perm[xx] = -1;

  	int[][] typePerms = new int[types.length][];
  	
  	for( int typei = 0; typei < types.length; typei++ ) {
  		String type = types[typei];
  		int[] roles = bestLearnedSchemaForMUCTemplate(numLearnedRoles, numTemplates, type);
  		for( int xx = 0; xx < types.length; xx++ )
  			if( roles[xx] > -1 )
  				perm[roles[xx]] = xx;
  		typePerms[typei] = roles;
  	}
  	
  	// Print them out for logging.
  	System.out.println("*** INITIAL BEST MUC TEMPLATE TO LEARNED SCHEMA MAPPING ***");
  	for( int typei = 0; typei < types.length; typei++ ) {
  		String type = types[typei];
  		System.out.println(type + "\t-> " + Arrays.toString(typePerms[typei]));
  	}
  	System.out.println("final mapping perm=" + Arrays.toString(perm));

  	// REMOVE BAD MAPPINGS!
  	// Some learned roles might map to a gold slot poorly. This hurts performance. 
  	// These are likely not true mappings to gold slots, so remove them.
  	int[] revisedPerm = new int[perm.length];
  	for( int xx = 0; xx < revisedPerm.length; xx++ ) revisedPerm[xx] = perm[xx];
  	double[] bestOverallPRF1 = evaluateSlotsIgnoringEventTypes(revisedPerm, false);
  	for( int xx = 0; xx < perm.length; xx++ ) {
  		if( revisedPerm[xx] > -1 ) {
  			revisedPerm[xx] = -1;
  			double[] PRF1 = evaluateSlotsIgnoringEventTypes(revisedPerm, false);
  			// If removing this mapping hurt performance, then it's likely a true mapping. Keep it. 
  			if( PRF1[2] < bestOverallPRF1[2]-.001 ) {
  				revisedPerm[xx] = perm[xx];
  			}
  		}
  	}
  	perm = revisedPerm;

  	// Print them out for logging.
  	System.out.println("*** BEST MUC TEMPLATE TO LEARNED SCHEMA MAPPING (AFTER TRIMMING) ***");
  	for( int typei = 0; typei < types.length; typei++ ) {
  		String type = types[typei];
  		System.out.println(type + "\t-> " + Arrays.toString(typePerms[typei]));
  	}
  	System.out.println("final mapping perm=" + Arrays.toString(perm));

		
  	return perm;
  }
  
  /**
   * This function evaluates a single slot, mapping a single learned role to a template slot.
   * Just give it the index of the gold slot, and the index of the learned slot we want to align in the evaluation.
   * @param sloti The domain's slot index.
   * @param rolei The learned slot index.
   * @param debug True if you want some debugging output.
   * @return Three values: precision, recall, F1
   */
  private double[] evaluateSlot(int sloti, int rolei, boolean debug) {
  	return evaluateSlot(sloti, rolei, null, debug);
  }
  	
  /**
   * Same as above, but if you give a template type, it only evaluates on that template's slot instead
   * of merging that slot across all template types.
   */
  private double[] evaluateSlot(int sloti, int rolei, String specificTemplateType, boolean debug) {
  	if( rolei == 208 || rolei == 210 ) // hardcode kidnap topics
  		debug = true;
  	
    System.out.println("evaluateSlot sloti= " + sloti + "\trolei=" + rolei);
    int unfilled = 0, unfilledCorrect = 0;
    // Initialize the counts of correct/incorrect/missed.
    int[] overall = new int[3];

    int docid = 0;
    for( String storyname : _docNames ) {
      if( debug ) System.out.println("story=" + storyname);
      List<Template> goldTemplates = _answerKey.getTemplates(storyname);
      if( debug ) System.out.println("  -> got " + (goldTemplates == null ? 0 : goldTemplates.size()) + " gold templates.");
      if( specificTemplateType != null ) {
      	goldTemplates = EvaluateTemplates.templatesContain(specificTemplateType, goldTemplates);
      	if( debug ) System.out.println("  -> selected " + (goldTemplates == null ? 0 : goldTemplates.size()) + " gold templates.");
      }
      List<TextEntity> entities = _docGuesses.get(docid++);

      if( debug && goldTemplates != null)
        for( Template plate : goldTemplates ) {
          System.out.println("plate class = " + plate.get(MUCTemplate.INCIDENT_TYPE));
          System.out.println(plate);
        }
      
      // No templates, so skip this document in the evaluation.
      if( _evaluateOnTemplateDocsOnly && goldTemplates == null ) { }

      // Nothing to do.
      else if( goldTemplates == null && entities == null ) {  }

      // No gold templates, these are all false positives.
      else if( goldTemplates == null ) {
      	// TEMPORARY - ONLY TEST ON THESE
      	if( !storyname.toLowerCase().startsWith("tst3") && !storyname.toLowerCase().startsWith("tst4") ) {
//      		System.out.println("skipping " + storyname);
      		continue;
      	}
      	
        if( debug ) System.out.println("No gold templates here.");
        if( sloti > -1 ) {
          if( debug ) System.out.println("Checking false positives for sloti " + sloti);
//          if( debug ) System.out.println("Entities: " + entities);
          // Count how many entities are labeled with this role.
          int falsePositives = 0;
          for( TextEntity entity : entities ) {
            if( entity.hasLabel(rolei) ) 
              falsePositives++;
          }
          // Increment the total false positive count.
          if( debug ) System.out.println("  Adding " + falsePositives + " false positives null template case.");
          overall[1] += falsePositives;
        }
      }

      // Gold templates, record the matches.
      else {
        // Gold answers for each slot.
        List<List<MUCEntity>> goldSlots = EvaluateTemplates.getAllGoldSlots(goldTemplates, _answerKey.numSlots());
        if( debug ) System.out.println("gold sloti " + sloti + ": " + goldSlots.get(sloti));

        // False negatives.
        if( entities == null ) {
          if( debug ) System.out.println("False negatives sloti=" + sloti);
          if( debug ) System.out.println("  Adding " + goldSlots.get(sloti).size() + " false negatives.");
          if( debug ) System.out.println("   - " + EvaluateTemplates.removeOptionals(goldSlots.get(sloti)).size() + " after removing optionals.");
          overall[2] += EvaluateTemplates.removeOptionals(goldSlots.get(sloti)).size();
        }

        // Compare all labeled entities to the gold slots.
        else {
          if( sloti > -1 ) {
            if( debug ) System.out.println("Comparing rolei=" + rolei + " sloti=" + sloti);

            // Get the entity string guesses we selected.
            List<String> guessedStrings = new ArrayList<String>();
            for( TextEntity entity : entities )
              if( entity.hasLabel(rolei) ) { 
                if( debug ) System.out.println("  guessed entity: " + entity);
                guessedStrings.add(entity.getCoreTokenRaw());
              }
            if( debug ) System.out.println("Guessed strings: " + guessedStrings);              

            // Evaluate the guesses with the golds!  (last parameter is DEBUG)
            int[] matches = TemplateTester.evaluateEntities(goldSlots.get(sloti), guessedStrings, false);
            if( debug ) System.out.println("   matches " + Arrays.toString(matches));
            for( int j = 0; j < overall.length; j++ ) overall[j] += matches[j];

            // Count how many slots we didn't guess anything for.
            if( matches[0] == 0 && matches[1] == 0 && matches[2] > 0 ) {
              if( matches[2] > 0)
                unfilled++;
              else unfilledCorrect++;
            }
          }
        }
      } // else
    } // story loop

    // Print the results!!
    System.out.println("evaluateSlot Results (sloti=" + sloti + " rolei=" + rolei + ")");
    System.out.println("  x = [correct, false-pos, false-neg]");
    System.out.println("  " + sloti + " = " + Arrays.toString(overall));
    
    // Debug output, per slot...
    float precision = ((float)overall[0] / (float)(overall[0]+overall[1]));
    float recall = ((float)overall[0] / (float)(overall[0]+overall[2]));
    float f1 = 2.0f * (precision*recall) / (precision+recall);
    System.out.printf("\tslot " + sloti + "\tprec=%.3f\trecall=%.3f\tf1=%.2f\n", precision, recall, f1);

    double[] scores = EvaluateTemplates.score(overall[0], overall[1], overall[0]+overall[2]);
    return scores;
  }
  
  /**
   * 
   * @param perm An array where each array index is a learned role, and its value is the MUC slot index for that learned role.
   * @return Array of doubles: precision, recall, F1
   */
//  public double[] evaluateSlotsIgnoringEventTypes(int[] perm, boolean debug) {
//    int[][] perms = new int[perm.length][];
//    for( int ii = 0; ii < perm.length; ii++ ) {
//      perms[ii] = new int[1];
//      perms[ii][0] = perm[ii];
//    }
//    return evaluateSlotsIgnoringEventTypes(perms, debug);
//  }
  
  public double[] evaluateSlotsIgnoringEventTypes(int[] perm, boolean debug) {
    System.out.println("evalSlotsIgnoringEventTypes perm=" + Arrays.toString(perm));
    int unfilled = 0, unfilledCorrect = 0;
    // Initialize the counts of correct/incorrect/missed.
    List<int[]> slotResults = new ArrayList<int[]>();
    for( int i = 0; i < _answerKey.numSlots(); i++ ) slotResults.add(new int[3]);

    int docid = 0;
    for( String storyname : _docNames ) {
      if( debug ) System.out.println("story=" + storyname);
      List<Template> goldTemplates = _answerKey.getTemplates(storyname);
      List<TextEntity> entities = _docGuesses.get(docid++);

      // No templates, so skip this document in the evaluation.
      if( _evaluateOnTemplateDocsOnly && goldTemplates == null ) { }
      
      // Nothing to do.
      else if( goldTemplates == null && entities == null ) {  }

      // No gold templates, these are all false positives.
      else if( goldTemplates == null ) {
      	// TEMPORARY - ONLY TEST ON THESE
      	if( !storyname.toLowerCase().startsWith("tst3") && !storyname.toLowerCase().startsWith("tst4") ) {
//      		System.out.println("skipping " + storyname);
      		continue;
      	}
      	
        if( debug ) System.out.println("No gold templates here.");
        for( int rolei = 0; rolei < _numLearnedSlots; rolei++ ) {
          int sloti = perm[rolei];
          if( sloti > -1 ) {
            if( debug ) System.out.println("False positives sloti=" + sloti);
            // Count how many entities are labeled with this role.
            int falsePositives = 0;
            for( TextEntity entity : entities ) {
              if( entity.hasLabel(rolei) ) 
                falsePositives++;
            }
            // Increment the total false positive count.
            int[] overall = slotResults.get(sloti);
            if( debug ) System.out.println("  Adding " + falsePositives + " false positives.");
            overall[1] += falsePositives;
          }
        }
      }

      // Gold templates, record the matches.
      else {
        // Gold answers for each slot.
        List<List<MUCEntity>> goldSlots = EvaluateTemplates.getAllGoldSlots(goldTemplates, _answerKey.numSlots());
        if( debug ) System.out.println("gold slots: ");
        if( debug ) for( List<MUCEntity> slot : goldSlots ) System.out.println("  slot: " + slot);

        // False negatives.
        if( entities == null ) {
          for( int sloti = 0; sloti < goldSlots.size(); sloti++ ) {
            if( debug ) System.out.println("False negatives sloti=" + sloti);
            if( debug ) System.out.println("  Adding " + goldSlots.get(sloti).size() + " false negatives.");
            if( debug ) System.out.println("   - " + EvaluateTemplates.removeOptionals(goldSlots.get(sloti)).size() + " after removing optionals.");
            int[] overall = slotResults.get(sloti);
            overall[2] += EvaluateTemplates.removeOptionals(goldSlots.get(sloti)).size();
          }
        }

        // Compare all labeled entities to the gold slots.
        else {
          Set<Integer> slotsChecked = new HashSet<Integer>();
          
          // NEW
          for( int sloti = 0; sloti < _answerKey.numSlots(); sloti++ ) {
            List<String> guessedStrings = new ArrayList<String>();
            for( int rolei = 0; rolei < _numLearnedSlots; rolei++ ) {
              if( perm[rolei] == sloti ) {
                for( TextEntity entity : entities ) {
                  if( entity.hasLabel(rolei) ) { 
                    if( debug ) System.out.println("  guessed entity: " + entity);
                    guessedStrings.add(entity.getCoreTokenRaw());
                  }
                }
                slotsChecked.add(sloti);
              }
            }
            if( debug ) System.out.println("Guessed strings: " + guessedStrings);          

            // Evaluate the guesses with the golds!  (last parameter is DEBUG)
            int[] matches = TemplateTester.evaluateEntities(goldSlots.get(sloti), guessedStrings, false);
            if( debug ) System.out.println("   matches " + Arrays.toString(matches));
            int[] overall = slotResults.get(sloti);
            for( int j = 0; j < overall.length; j++ ) overall[j] += matches[j];

            // Count how many slots we didn't guess anything for.
            if( matches[0] == 0 && matches[1] == 0 && matches[2] > 0 ) {
              if( matches[2] > 0)
                unfilled++;
              else unfilledCorrect++;
            }
          }
          
          // OLD 
//          for( int rolei = 0; rolei < _numLearnedSlots; rolei++ ) {
//            int sloti = perm[rolei];
//            if( sloti > -1 ) {
//              if( debug ) System.out.println("Comparing rolei=" + rolei + " sloti=" + sloti);
//              slotsChecked.add(sloti);
//
//              // Get the entity string guesses we selected.
//              List<String> guessedStrings = new ArrayList<String>();
//              for( TextEntity entity : entities )
//                if( entity.getLabel() == rolei ) { 
//                  if( debug ) System.out.println("  guessed entity: " + entity);
//                  guessedStrings.add(entity.getCoreTokenRaw());
//                }
//              if( debug ) System.out.println("Guessed strings: " + guessedStrings);              
//              
//              // Evaluate the guesses with the golds!  (last parameter is DEBUG)
//              int[] matches = TemplateTester.evaluateEntities(goldSlots.get(sloti), guessedStrings, false);
//              if( debug ) System.out.println("   matches " + Arrays.toString(matches));
//              int[] overall = slotResults.get(sloti);
//              for( int j = 0; j < overall.length; j++ ) overall[j] += matches[j];
//
//              // Count how many slots we didn't guess anything for.
//              if( matches[0] == 0 && matches[1] == 0 && matches[2] > 0 ) {
//                if( matches[2] > 0)
//                  unfilled++;
//                else unfilledCorrect++;
//              }
//            }
//          }
          
          
          // NEW 09/12/2012 : Any slots not mapped are now false negatives.
          for( int sloti = 0; sloti < goldSlots.size(); sloti++ ) {
            if( !slotsChecked.contains(sloti) ) {
              int[] overall = slotResults.get(sloti);
              overall[2] += EvaluateTemplates.removeOptionals(goldSlots.get(sloti)).size();
            }
          }
        }
      } // else
    } // story loop

    // Print the results!!
    System.out.println("evaluateSlotsIgnoringEventTypes Results (perm=" + Arrays.toString(perm) + ")");
    System.out.print("evaluateSlotsIgnoringEventTypes Results (perm=");
    for( int xx = 0; xx < perm.length; xx++ ) System.out.print(xx + ":" + perm[xx] + " ");
    System.out.println(")");
    System.out.println("  x = [correct, false-pos, false-neg]");
    for( int i = 0; i < slotResults.size(); i++ )
      System.out.println("  " + i + " = " + Arrays.toString(slotResults.get(i)));
    
    // Calculate the overall F1 score.
    int[] allscores = new int[slotResults.get(0).length];
    for( int i = 0; i < slotResults.size(); i++ ) {
      int[] scores = slotResults.get(i);
      
      if( scores[0] + scores[2] > 0 ) {// don't count slots that had no gold entities. any of our guesses are thus ignored
        for( int j = 0; j < scores.length; j++ )
          allscores[j] += scores[j];
      }

      // Debug output, per slot...
      float precision = ((float)scores[0] / (float)(scores[0]+scores[1]));
      float recall = ((float)scores[0] / (float)(scores[0]+scores[2]));
      float f1 = 2.0f * (precision*recall) / (precision+recall);
      System.out.printf("\tslot " + i + "\tprec=%.3f\trecall=%.3f\tf1=%.2f\n", precision, recall, f1);
    }
    double[] scores = EvaluateTemplates.score(allscores[0], allscores[1], allscores[0]+allscores[2]);
    System.out.printf("\tall\tprec=%.3f\trecall=%.3f\tf1=%.2f\n", scores[0], scores[1], scores[2]);

    return scores;
  }

  
  public static void main(String[] args) {
  }
}
