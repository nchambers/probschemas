package nate.probschemas;

import java.io.Serializable;
import java.util.Arrays;

import edu.stanford.nlp.stats.ClassicCounter;
import edu.stanford.nlp.util.ErasureUtils;

public class EntityModelInstance implements Serializable {
  static final long serialVersionUID = 10000;

  public double likelihood;
  public int samplingStep;
  // Model.
//  public int[][][] words; // [doc][entity][mention]
//  public int[][][] deps;  // [doc][entity][mention]
//  public int[][][] verbs; // [doc][entity][mention]
//  public int[][][] inverseDeps;  // [doc][entity][mention]   : the inverse of "subj" is "obj" and vice versa
//  public int[][][] feats;  // [doc][entity][feattype]
//  
  public int[][] zs;      // [doc][entity]  zs are z variable assignments to single entities

  // Required.
  public double[] topicCounts;      // when one global Theta distribution
  public int[][] topicCountsByDoc;  // when Thetas are per document
  public ClassicCounter<Integer>[] wCountsBySlot;
  public ClassicCounter<Integer>[] verbCountsBySlot;
  public ClassicCounter<Integer>[] depCountsBySlot;
  public ClassicCounter<Integer>[] featCountsBySlot;


  public EntityModelInstance() {
    // TODO Auto-generated constructor stub
  }

  public void storeAll(
      int[][] zs,
      double[] topicCounts, 
      int[][] topicCountsByDoc,
      ClassicCounter<Integer>[] wCountsBySlot, 
      ClassicCounter<Integer>[] verbCountsBySlot,
      ClassicCounter<Integer>[] depCountsBySlot,
      ClassicCounter<Integer>[] featCountsBySlot) {

    this.zs = new int[zs.length][];
    for( int xx = 0; xx < zs.length; xx++ )
      this.zs[xx] = Arrays.copyOf(zs[xx], zs[xx].length);
    
    this.topicCounts = Arrays.copyOf(topicCounts, topicCounts.length);
    this.topicCountsByDoc = new int[topicCountsByDoc.length][];
    for( int xx = 0; xx < topicCountsByDoc.length; xx++ )
      this.topicCountsByDoc[xx] = Arrays.copyOf(topicCountsByDoc[xx], topicCountsByDoc[xx].length);
    
    this.wCountsBySlot = cloneCounter(wCountsBySlot);
    this.verbCountsBySlot = cloneCounter(verbCountsBySlot);
    this.depCountsBySlot = cloneCounter(depCountsBySlot);
    this.featCountsBySlot = cloneCounter(featCountsBySlot);
  }
  
  /**
   * Make a copy of the array of counters.
   */
  public ClassicCounter<Integer>[] cloneCounter(ClassicCounter<Integer>[] counter) {
    ClassicCounter<Integer>[] newcount = ErasureUtils.<ClassicCounter<Integer>>mkTArray(ClassicCounter.class, counter.length);
    for( int xx = 0; xx < counter.length; xx++ ) {
      ClassicCounter<Integer> cc = new ClassicCounter<Integer>();
      newcount[xx] = cc;
      for( Integer key : counter[xx].keySet() )
        cc.incrementCount(key, counter[xx].getCount(key));
    }
    return newcount;
  }
  
}
