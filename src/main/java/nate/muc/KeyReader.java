package nate.muc;

import java.util.List;

public interface KeyReader {
  public List<Template> getTemplates(String storyName);
  public int numSlots();
}
