package nate;

import java.util.List;

import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.trees.TypedDependency;
import nate.util.TreeOperator;
import nate.EntityMention;
import nate.NERSpan;

public class ProcessedDocument {
  public String storyname;
  public List<String> parses;
  public List<NERSpan> ners;
  public List<EntityMention> mentions;
  public List<List<TypedDependency>> deps;

  public ProcessedDocument(String name, List<String> parses, List<List<TypedDependency>> deps, List<EntityMention> mentions, List<NERSpan> ners) {
    this.storyname = name;
    this.parses = parses;
    this.deps = deps;
    this.mentions = mentions;
    this.ners = ners;
  }
  
  public List<Tree> trees() {
    if( parses != null )
      return TreeOperator.stringsToTrees(parses);
    else return null;
  }
}
