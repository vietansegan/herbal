package data;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.util.ArrayList;
import java.util.HashMap;
import util.DataUtils;
import util.IOUtils;

/**
 *
 * @author vietan
 */
public class Debate extends LabelTextDataset {

    // inputs
    private ArrayList<String> speakerList;
    private ArrayList<String> billList;
    // flat turns
    private int[] speakers;
    private String[] bills;
    // processed data
    private String[][] debateTurnIds;
    private int[][] debateTurnLabels; // set of labels for each debate
    private int[][][] debateTurnWords;
    private int[][] debateTurnSpeakers;
    private ArrayList<String> speakerVocab;

    public Debate(String name, String folder) {
        super(name, folder);
    }

    public Debate(String name, String folder,
            CorpusProcessor corpProc) {
        super(name, folder, corpProc);
    }

    public int[] getSpeakers() {
        return this.speakers;
    }

    public String[] getBills() {
        return this.bills;
    }

    public int[][] getDebateTurnSpeakers() {
        return this.debateTurnSpeakers;
    }

    public int[][] getDebateLabels() {
        return this.debateTurnLabels;
    }

    public int[][][] getDebateTurnWords() {
        return this.debateTurnWords;
    }

    public ArrayList<String> getSpeakerList() {
        return this.speakerList;
    }

    public ArrayList<String> getBillList() {
        return this.billList;
    }

    public void loadSpeakers(File speakerFile) throws Exception {
        logln("--- Loading turn speakers from " + speakerFile);
        if (this.docIdList == null) {
            throw new RuntimeException("List of document ids has not been loaded");
        }

        HashMap<String, String> docSpeakerMap = new HashMap<String, String>();
        BufferedReader reader = IOUtils.getBufferedReader(speakerFile);
        int count = 0;
        String line;
        String[] sline;
        while ((line = reader.readLine()) != null) {
            sline = line.split("\t");
            String docId = sline[0];
            String speakerId = sline[1];
            docSpeakerMap.put(docId, speakerId);
            count++;
        }
        reader.close();
        logln("--- Loaded. # turn speakers: " + count);

        this.speakerList = new ArrayList<String>();
        for (String docId : this.docIdList) {
            String speaker = docSpeakerMap.get(docId);
            this.speakerList.add(speaker);
        }
    }

    public void loadBills(File billFile) throws Exception {
        logln("--- Loading turn bills from " + billFile);
        if (this.docIdList == null) {
            throw new RuntimeException("List of document ids has not been loaded");
        }

        HashMap<String, String> docBillMap = new HashMap<String, String>();
        BufferedReader reader = IOUtils.getBufferedReader(billFile);
        int count = 0;
        String line;
        String[] sline;
        while ((line = reader.readLine()) != null) {
            sline = line.split("\t");
            String docId = sline[0];
            String billId = sline[1];
            docBillMap.put(docId, billId);
            count++;
        }
        reader.close();
        logln("--- Loaded. # turn speakers: " + count);

        this.billList = new ArrayList<String>();
        for (String docId : this.docIdList) {
            String speaker = docBillMap.get(docId);
            this.billList.add(speaker);
        }
    }

    public ArrayList<String> getDebateTextList() {
        HashMap<String, ArrayList<Integer>> debateTurnIndices = new HashMap<String, ArrayList<Integer>>();
        for (int ii = 0; ii < this.docIdList.size(); ii++) {
            String turnId = docIdList.get(ii);
            String billId = billList.get(ii);

            String debateId = turnId.split("_")[0];
            if (!billId.equals("null")) {
                debateId = debateId + "_" + billId;
            }

            ArrayList<Integer> turnIndices = debateTurnIndices.get(debateId);
            if (turnIndices == null) {
                turnIndices = new ArrayList<Integer>();
            }
            turnIndices.add(ii);
            debateTurnIndices.put(debateId, turnIndices);
        }

        ArrayList<String> debateTextList = new ArrayList<String>();
        for (String debateId : debateTurnIndices.keySet()) {
            StringBuilder str = new StringBuilder();
            ArrayList<Integer> turnIndices = debateTurnIndices.get(debateId);
            for (int idx : turnIndices) {
                str.append(textList.get(idx)).append("\n");
            }
            debateTextList.add(str.toString().trim());
        }

        return debateTextList;
    }

    @Override
    public void format(String outputFolder) throws Exception {
        IOUtils.createFolder(outputFolder);

        if (speakerVocab == null) {
            createSpeakerVocab();
        }

        File speakerVocFile = new File(outputFolder, formatFilename + speakerVocabExt);
        logln("--- Outputing label vocab ... " + speakerVocFile.getAbsolutePath());
        DataUtils.outputVocab(speakerVocFile.getAbsolutePath(),
                this.speakerVocab);

        // get speaker indices
        this.speakers = new int[this.speakerList.size()];
        for (int ii = 0; ii < speakers.length; ii++) {
            int speakerIdx = speakerVocab.indexOf(speakerList.get(ii));
            this.speakers[ii] = speakerIdx;
        }

        // perform normal processing
        super.format(outputFolder);
    }

    public void setSpeakerVocab(ArrayList<String> speakerVoc) {
        this.speakerVocab = speakerVoc;
    }

    public ArrayList<String> getSpeakerVocab() {
        return this.speakerVocab;
    }

    private void createSpeakerVocab() {
        this.speakerVocab = new ArrayList<String>();
        for (String speaker : speakerList) {
            if (!this.speakerVocab.contains(speaker)) {
                speakerVocab.add(speaker);
            }
        }
    }

    @Override
    protected void outputDocumentInfo(String outputFolder) throws Exception {
        File outputFile = new File(outputFolder, formatFilename + docInfoExt);
        logln("--- Outputing document info ... " + outputFile);

        BufferedWriter infoWriter = IOUtils.getBufferedWriter(outputFile);
        for (int docIndex : this.processedDocIndices) {
            infoWriter.write(this.docIdList.get(docIndex)
                    + "\t" + this.speakers[docIndex]
                    + "\t" + this.billList.get(docIndex));
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
        ArrayList<Integer> speakerIndexList = new ArrayList<Integer>();
        billList = new ArrayList<String>();
        ArrayList<int[]> labelIndexList = new ArrayList<int[]>();
        while ((line = reader.readLine()) != null) {
            sline = line.split("\t");
            docIdList.add(sline[0]);
            speakerIndexList.add(Integer.parseInt(sline[1]));
            billList.add(sline[2]);
            int[] labelIndices = new int[sline.length - 3];
            for (int ii = 0; ii < sline.length - 3; ii++) {
                labelIndices[ii] = Integer.parseInt(sline[ii + 3]);
            }
            labelIndexList.add(labelIndices);
        }
        reader.close();

        this.docIds = docIdList.toArray(new String[docIdList.size()]);
        this.labels = new int[labelIndexList.size()][];

        for (int ii = 0; ii < this.labels.length; ii++) {
            this.labels[ii] = labelIndexList.get(ii);
        }

        this.speakers = new int[speakerIndexList.size()];
        for (int ii = 0; ii < speakerIndexList.size(); ii++) {
            this.speakers[ii] = speakerIndexList.get(ii);
        }

        this.bills = billList.toArray(new String[billList.size()]);
    }

    @Override
    public void loadFormattedData(String fFolder) {
        try {
            super.loadFormattedData(fFolder);
            this.inputSpeakerVocab(new File(fFolder, formatFilename + speakerVocabExt));
            this.getDebateTurnData();
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(1);
        }
    }

    private void getDebateTurnData() {
        HashMap<String, ArrayList<Integer>> debateTurnIndices = new HashMap<String, ArrayList<Integer>>();
        for (int ii = 0; ii < docIds.length; ii++) {
            String turnId = docIds[ii];
            String billId = bills[ii];
            String debateId = turnId.split("_")[0];
            if (!billId.equals("null")) {
                debateId = debateId + "_" + billId;
            }

            ArrayList<Integer> turnIndices = debateTurnIndices.get(debateId);
            if (turnIndices == null) {
                turnIndices = new ArrayList<Integer>();
            }
            turnIndices.add(ii);
            debateTurnIndices.put(debateId, turnIndices);
        }

        int D = debateTurnIndices.size();
        this.debateTurnIds = new String[D][];
        this.debateTurnWords = new int[D][][];
        this.debateTurnLabels = new int[D][];
        this.debateTurnSpeakers = new int[D][];
        int d = 0;
        for (String dId : debateTurnIndices.keySet()) {
            ArrayList<Integer> turnIndices = debateTurnIndices.get(dId);

            // validate: whether the sets of labels for all turns in this debate
            // are the same
            int[] firstTurnLabels = labels[turnIndices.get(0)];
            for (int ii = 1; ii < turnIndices.size(); ii++) {
                if (!compareArray(firstTurnLabels, labels[turnIndices.get(ii)])) {
                    throw new RuntimeException("Labels of different turns in the "
                            + "same debate are different.");
                }
            }

            this.debateTurnLabels[d] = firstTurnLabels;
            int Td = turnIndices.size();
            this.debateTurnIds[d] = new String[Td];
            this.debateTurnWords[d] = new int[Td][];
            this.debateTurnSpeakers[d] = new int[Td];
            for (int t = 0; t < Td; t++) {
                int idx = turnIndices.get(t);
                this.debateTurnIds[d][t] = docIds[idx];
                this.debateTurnWords[d][t] = words[idx];
                this.debateTurnSpeakers[d][t] = speakers[idx];
            }
            d++;
        }
    }

    private boolean compareArray(int[] firstArray, int[] secondArray) {
        if (firstArray.length != secondArray.length) {
            return false;
        }
        for (int ii = 0; ii < firstArray.length; ii++) {
            if (firstArray[ii] != secondArray[ii]) {
                return false;
            }
        }
        return true;
    }

    protected void inputSpeakerVocab(File file) throws Exception {
        speakerVocab = new ArrayList<String>();
        BufferedReader reader = IOUtils.getBufferedReader(file);
        String line;
        while ((line = reader.readLine()) != null) {
            speakerVocab.add(line);
        }
        reader.close();
    }
}
