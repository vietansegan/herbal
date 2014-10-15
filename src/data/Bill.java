package data;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Set;
import util.DataUtils;
import util.IOUtils;

/**
 *
 * @author vietan
 */
public class Bill extends LabelTextDataset {

    public static final String topicVocabExt = ".tvoc";
    public static final String topicExt = ".topics";
    // raw topics
    protected HashMap<String, String> topicMap;

    // formatted topics
    protected ArrayList<String> topicVocab;
    protected int[] topics;

    public Bill(String name, String folder) {
        super(name, folder);
    }

    public Bill(String name, String folder,
            CorpusProcessor corpProc) {
        super(name, folder, corpProc);
    }

    public ArrayList<String> getTopicVocab() {
        return this.topicVocab;
    }

    public int[] getTopics() {
        return this.topics;
    }

    public void loadTopics(File filepath) throws Exception {
        BufferedReader reader = IOUtils.getBufferedReader(filepath);
        this.topicMap = new HashMap<>();
        String line;
        String[] sline;
        while ((line = reader.readLine()) != null) {
            sline = line.split("\t");

            String docId = sline[0];
            String topic = sline[3];
            if (topic.equals("null")) {
                topic = "Other";
            }
            this.topicMap.put(docId, topic);
        }
        reader.close();
    }

    @Override
    public void loadFormattedData(String fFolder) {
        try {
            super.loadFormattedData(fFolder);
            this.inputTopicVocab(new File(fFolder, formatFilename + topicVocabExt));
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while loading formatted data "
                    + "from " + fFolder);
        }
    }

    protected void inputTopicVocab(File file) throws Exception {
        topicVocab = new ArrayList<String>();
        BufferedReader reader = IOUtils.getBufferedReader(file);
        String line;
        while ((line = reader.readLine()) != null) {
            topicVocab.add(line);
        }
        reader.close();
    }

    @Override
    public void format(String outputFolder) throws Exception {
        IOUtils.createFolder(outputFolder);
        formatTopic(outputFolder);
        super.format(outputFolder);
    }

    protected void formatTopic(String outputFolder) throws Exception {
        Set<String> topicSet = new HashSet<>();
        for (String topic : topicMap.values()) {
            topicSet.add(topic);
        }
        topicVocab = new ArrayList<>();
        for (String topic : topicSet) {
            topicVocab.add(topic);
        }
        Collections.sort(topicVocab);

        File topicVocFile = new File(outputFolder, formatFilename + topicVocabExt);
        logln("--- Outputing topic vocab ... " + topicVocFile.getAbsolutePath());
        DataUtils.outputVocab(topicVocFile.getAbsolutePath(), this.topicVocab);
    }

    @Override
    protected void outputDocumentInfo(String outputFolder) throws Exception {
        File outputFile = new File(outputFolder, formatFilename + docInfoExt);
        logln("--- Outputing document info ... " + outputFile);

        BufferedWriter infoWriter = IOUtils.getBufferedWriter(outputFile);
        for (int docIndex : this.processedDocIndices) {
            String docId = docIdList.get(docIndex);
            infoWriter.write(docId);

            // topic
            int topicIdx = Collections.binarySearch(topicVocab, topicMap.get(docId));
            infoWriter.write("\t" + topicIdx);

            // labels
            for (int label : labels[docIndex]) {
                infoWriter.write("\t" + label);
            }
            infoWriter.write("\n");
        }
        infoWriter.close();
    }

    @Override
    public void inputDocumentInfo(File file) throws Exception {
        logln("--- Reading document info from " + file);

        BufferedReader reader = IOUtils.getBufferedReader(file);
        String line;
        String[] sline;
        docIdList = new ArrayList<String>();
        ArrayList<Integer> topicIndexList = new ArrayList<>();
        ArrayList<int[]> labelIndexList = new ArrayList<int[]>();
        while ((line = reader.readLine()) != null) {
            sline = line.split("\t");
            docIdList.add(sline[0]);
            topicIndexList.add(Integer.parseInt(sline[1]));
            int[] labelIndices = new int[sline.length - 2];
            for (int ii = 0; ii < sline.length - 2; ii++) {
                labelIndices[ii] = Integer.parseInt(sline[ii + 2]);
            }
            labelIndexList.add(labelIndices);
        }
        reader.close();

        this.docIds = docIdList.toArray(new String[docIdList.size()]);
        this.topics = new int[topicIndexList.size()];
        for (int ii = 0; ii < this.topics.length; ii++) {
            this.topics[ii] = topicIndexList.get(ii);
        }
        this.labels = new int[labelIndexList.size()][];
        for (int ii = 0; ii < this.labels.length; ii++) {
            this.labels[ii] = labelIndexList.get(ii);
        }
    }
}
