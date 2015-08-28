package nate.probschemas;

public interface Sampler {
  public void runSampler(int numIterations);
  public void printWordDistributionsPerTopic();
  public void toFile(String file);
//  public Sampler fromFile(String file);
}
