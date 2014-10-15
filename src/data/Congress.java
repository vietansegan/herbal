package data;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import util.IOUtils;
import util.freedomworks.FWLegislator;
import util.freedomworks.FWYear;
import util.govtrack.GTLegislator;
import util.govtrack.GTProcessorV2;

/**
 *
 * @author vietan
 */
public class Congress extends AbstractTokenizeDataset {

    public static final String BILL_PREFIX = "bill-";
    public static final String DEBATE_PREFIX = "debate-";
    private final Bill billData;
    private final Debate debateData;
    private HashMap<String, GTLegislator> legislators;
    private ArrayList<String> labelVocab;
    private FWYear[] congressYears;

    public Congress(String datasetName, String folder) {
        super(datasetName, folder);
        this.billData = new Bill(BILL_PREFIX + this.name, folder);
        this.debateData = new Debate(DEBATE_PREFIX + this.name, folder);
    }

    public Congress(String datasetName, String folder, CorpusProcessor corpProc) {
        super(datasetName, folder, corpProc);
        this.billData = new Bill(BILL_PREFIX + this.name, folder, corpProc);
        this.debateData = new Debate(DEBATE_PREFIX + this.name, folder, corpProc);
    }

    public void setCongressYear(FWYear[] y) {
        this.congressYears = y;
    }

    public FWYear[] getCongressYears() {
        return this.congressYears;
    }

    public GTLegislator getLegislator(String govtrackId) {
        return this.legislators.get(govtrackId);
    }

    public HashMap<String, GTLegislator> getLegislators() {
        return this.legislators;
    }

    public void setLegislators(HashMap<String, GTLegislator> l) {
        this.legislators = l;
    }

    public Bill getBillData() {
        return this.billData;
    }

    public Debate getDebateData() {
        return this.debateData;
    }

    public ArrayList<String> getWordVocab() {
        return this.billData.getWordVocab();
    }

    public ArrayList<String> getLabelVocab() {
        return this.labelVocab;
    }

    public void setLabelVocab(ArrayList<String> labelVocab) {
        this.labelVocab = labelVocab;
    }

    public void loadData(String billFolder, String debateFolder) {
        try {
            if (billFolder != null) {
                billData.loadTextDataFromFolder(new File(billFolder, "texts"));
                billData.loadLabels(new File(billFolder, "subjects.txt")); // load labels
                billData.loadTopics(new File(billFolder, "topics.txt")); // load topics
            }
            if (debateFolder != null) {
                debateData.loadTextDataFromFolder(new File(debateFolder, "texts"));
                debateData.loadLabels(new File(debateFolder, "subjects.txt"));
                debateData.loadSpeakers(new File(debateFolder, "speakers.txt"));
                debateData.loadBills(new File(debateFolder, "bills.txt"));
            }
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while loading data.");
        }
    }

    public void loadLegislatorData(
            String legislatorData,
            String houseRepublicanFile,
            String senateRepublicanFile) {
        GTProcessorV2 proc = new GTProcessorV2();
        try {
            proc.inputLegislators(legislatorData);

            if (houseRepublicanFile != null) {
                proc.loadTeaPartyHouse(houseRepublicanFile);
            }

            if (senateRepublicanFile != null) {
                proc.loadTeaPartySenate(senateRepublicanFile);
            }
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while loading legislator data.");
        }

        legislators = proc.getLegislators();
    }

    public void loadLabelVocab(File labelVocFile) {
        this.labelVocab = new ArrayList<String>();
        try {
            BufferedReader reader = IOUtils.getBufferedReader(labelVocFile);
            String line;
            while ((line = reader.readLine()) != null) {
                this.labelVocab.add(line);
            }
            reader.close();
        } catch (IOException e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while loading label vocab. "
                    + labelVocFile);
        }
    }

    public void format(File outputFolder) throws Exception {
        format(outputFolder.getAbsolutePath());
    }

    public void format(String outputFolder) throws Exception {
        logln("Formatting ...");
        IOUtils.createFolder(outputFolder);

        ArrayList<String> billTextList = billData.textList;
        String[] billTexts = billTextList.toArray(new String[billTextList.size()]);
        logln("--- # bill texts: " + billTexts.length);

        ArrayList<String> debateTextList = debateData.getDebateTextList();
        String[] debateTexts = debateTextList.toArray(new String[debateTextList.size()]);
        logln("--- # debate texts: " + debateTexts.length);

        String[] rawTexts = concat(billTexts, debateTexts);
        corpProc.setRawTexts(rawTexts);
        corpProc.process();

        if (labelVocab != null) {
            debateData.setLabelVocab(labelVocab);
        }
        debateData.format(outputFolder);

        if (labelVocab != null) {
            billData.setLabelVocab(labelVocab);
        }
        billData.format(outputFolder);
    }

    public void loadFormattedData(File fFolder) {
        loadFormattedData(fFolder.getAbsolutePath());
    }

    public void loadFormattedData(String fFolder) {
        billData.loadFormattedData(fFolder);
        debateData.loadFormattedData(fFolder);
        labelVocab = billData.getLabelVocab();
    }

    public static <T> T[] concat(T[] first, T[] second) {
        T[] result = Arrays.copyOf(first, first.length + second.length);
        System.arraycopy(second, 0, result, first.length, second.length);
        return result;
    }

    public static Congress loadProcessedCongress(
            String congressNum,
            String datasetFolder,
            String billFolder,
            String debateFolder,
            String legislatorFile,
            String houseRepFile,
            String senateRepFile) {
        Congress congress = new Congress(congressNum, datasetFolder);
        congress.loadData(billFolder, debateFolder);
        congress.loadLegislatorData(legislatorFile, houseRepFile, senateRepFile);
        return congress;
    }

    public static Congress loadProcessedCongress(
            String congressNum,
            String datasetFolder,
            String billFolder,
            String debateFolder,
            String legislatorFile,
            String houseRepFile,
            String senateRepFile,
            CorpusProcessor corp) {
        Congress congress = new Congress(congressNum, datasetFolder, corp);
        congress.loadData(billFolder, debateFolder);
        congress.loadLegislatorData(legislatorFile, houseRepFile, senateRepFile);
        return congress;
    }

    public static void getFreedomWorksScore(
            HashMap<String, GTLegislator> gtLegislators,
            FWYear fwYear, String scoreType) {
        if (verbose) {
            logln("\nMatch and assign FreedomWorks score ...");
        }
        HashMap<Integer, FWLegislator> fwLegislators = fwYear.getLegislators();
        int gtCount = 0;
        HashMap<String, ArrayList<GTLegislator>> gtHash = new HashMap<String, ArrayList<GTLegislator>>();
        for (GTLegislator gtLeg : gtLegislators.values()) {
            if (gtLeg.getType().equals("sen")) { // exlude senators
                continue;
            }

            String lastname = gtLeg.getLastname();
            String state = gtLeg.getState();
            String type = gtLeg.getParty();

            String key = lastname + ":-:" + state;
            switch (type) {
                case "Republican":
                    key += ":-:R";
                    break;
                case "Democrat":
                    key += ":-:D";
                    break;
                default:
                    continue;
            }

            ArrayList<GTLegislator> list = gtHash.get(key);
            if (list == null) {
                list = new ArrayList<GTLegislator>();
            }
            list.add(gtLeg);
            gtHash.put(key, list);
            gtCount++;
        }

        int fwCount = 0;
        HashMap<String, ArrayList<FWLegislator>> fwHash = new HashMap<String, ArrayList<FWLegislator>>();
        for (FWLegislator fwLeg : fwLegislators.values()) {
            if (fwYear.getLegislatorScore(fwLeg.getId()) == FWYear.NA_SCORE) {
                continue;
            }

            String fwName = fwLeg.getProperty(FWLegislator.NAME);
            String role = fwLeg.getProperty(FWLegislator.ROLE);
            String state = role.substring(0, 2);
            String type = role.substring(role.length() - 1);

            if (!type.equals("R") && !type.equals("D")) {
                logln("Skipping " + fwLeg.toString() + ". Neither R nor D.");
            }

            String key = fwName + ":-:" + state + ":-:" + type;
            ArrayList<FWLegislator> list = fwHash.get(key);
            if (list == null) {
                list = new ArrayList<FWLegislator>();
            }
            list.add(fwLeg);
            fwHash.put(key, list);
            fwCount++;
        }

        System.out.println("gtcount = " + gtCount + "\tgtHash size = " + gtHash.size());
        System.out.println("fwcount = " + fwCount + "\tfwHash size = " + fwHash.size());

        int matchCount = 0;
        int nullCount = 0;
        int otmCount = 0;
        for (String key : gtHash.keySet()) {
            ArrayList<GTLegislator> gtList = gtHash.get(key);
            ArrayList<FWLegislator> fwList = fwHash.get(key);

            if (fwList == null) {
                nullCount++;
                continue;
            }

            if (gtList.size() != fwList.size()) {
                System.out.println(gtList.toString());
                System.out.println(fwList.toString());
                System.out.println("Different sizes");
                otmCount++;
            }

            if (gtList.size() == 1) {
                int fwId = fwList.get(0).getId();
                int score = fwYear.getLegislatorScore(fwId);
                gtList.get(0).addProperty(scoreType, Integer.toString(score));
                gtList.get(0).addProperty(GTLegislator.FW_ID, Integer.toString(fwId));
                matchCount++;
            } else {
                for (GTLegislator gtl : gtList) {
                    int gtDist = gtl.getDistrict();

                    for (FWLegislator fwl : fwList) {
                        String role = fwl.getProperty(FWLegislator.ROLE);
                        int fwDist = Integer.parseInt(role.substring(3, 5).trim());
                        if (gtDist == fwDist) {
                            int score = fwYear.getLegislatorScore(fwl.getId());
                            gtl.addProperty(scoreType, Integer.toString(score));
                            gtl.addProperty(GTLegislator.FW_ID, Integer.toString(fwl.getId()));
                            matchCount++;
                            break;
                        }
                    }
                }
            }
        }

        System.out.println("no score count = " + nullCount);
        System.out.println("match count = " + matchCount);
        System.out.println("different sizes = " + otmCount);
        System.out.println("size = " + gtLegislators.size());
    }
}
