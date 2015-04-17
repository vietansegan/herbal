package experiment.percongress;

import core.AbstractExperiment;
import core.AbstractModel;
import core.AbstractSampler;
import data.Author;
import data.AuthorVoteTextDataset;
import data.Bill;
import data.Congress;
import data.CorpusProcessor;
import data.Debate;
import data.TextDataset;
import data.Vote;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.IOException;
import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Date;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import labelestimator.TFEstimator;
import org.apache.commons.cli.BasicParser;
import org.apache.commons.cli.Options;
import sampler.unsupervised.LDA;
import util.CLIUtils;
import util.IOUtils;
import util.SparseVector;
import util.evaluation.Measurement;
import util.freedomworks.FWBill;
import util.freedomworks.FWVote;
import util.freedomworks.FWYear;
import util.govtrack.GTLegislator;
import votepredictor.AbstractVotePredictor;
import votepredictor.BayesianIdealPoint;
import votepredictor.IdealPoint;
import votepredictor.RandomPredictor;
import votepredictor.SLDAIdealPoint;
import votepredictor.SNLDAIdealPoint;
import votepredictor.SNHDPIdealPoint;
import sampling.likelihood.CascadeDirMult.PathAssumption;
import sampling.util.SparseCount;
import util.MiscUtils;
import util.MismatchRuntimeException;
import util.RankingItem;
import util.RankingItemList;
import util.StatUtils;
import util.normalizer.MinMaxNormalizer;
import votepredictor.BayesianMultIdealPoint;
import votepredictor.BayesianMultIdealPointOWLQN;
import votepredictor.LexicalIdealPoint;
import votepredictor.LexicalSNLDAIdealPoint;
import votepredictor.SLDAMultIdealPoint;
import votepredictor.textidealpoint.flat.HybridSLDAIdealPoint;
import votepredictor.textidealpoint.HybridSNHDPIdealPoint;
import votepredictor.textidealpoint.HybridSNLDAIdealPoint;
import votepredictor.textidealpoint.RecursiveSLDAIdealPoint;
import votepredictor.textidealpoint.flat.LexicalSLDAIdealPoint;
import votepredictor.textidealpoint.flat.HybridSLDAMultipleIdealPoint;
import votepredictor.textidealpoint.hierarchy.HierMultSHDP;
import votepredictor.textidealpoint.hierarchy.HierMultiTIPM;
import votepredictor.textidealpoint.hierarchy.HierSingleTIPM;
import votepredictor.textidealpoint.hierarchy.MultTopicIdealPoint;

/**
 *
 * @author vietan
 */
public class VotePredExpt extends AbstractExperiment<Congress> {

    public static final String GovtrackUrl = "https://www.govtrack.us/data/us/";
    public static final String AssignmentFile = "assignments.zip";
    public static final String AuthorScoreFile = "authors.score";
    public static final String VoteScoreFile = "votes.score";
    public static final String PRE_SCORE = "pre-score";
    public static final String POS_SCORE = "pos-score";
    public static final String AuthorErrorFile = "author-error.txt";
    public static final String LexicalFile = "lexical.txt";
    public static final int AGAINST = 0;
    public static final int WITH = 1;
    protected String congressNum;
    protected String datasetFolder;
    protected String formatFolder;
    protected String modelFolder;
    protected int numTopWords;
    protected String processedDataFolder;
    protected int numFolds;
    protected double trToDevRatio;
    protected double teRatio;
    public static HashMap<String, int[]> congressYearMap;
    protected HashMap<String, Vote> keyvotes;
    protected HashMap<Integer, HashMap<String, FWVote>> voteMap;
    protected FWYear[] fwYears;
    protected AuthorVoteTextDataset debateVoteData;
    protected Bill billData;
    protected int[][] votes;
    // train
    protected ArrayList<Integer> trainAuthorIndices;
    protected ArrayList<Integer> trainBillIndices;
    protected ArrayList<Integer> trainDebateIndices;
    protected boolean[][] trainVotes;
    protected int[][] trainVoteWords;
    protected int[] trainVoteTopics;
    // dev
    protected boolean[][] devVotes;
    // test
    protected ArrayList<Integer> testAuthorIndices;
    protected ArrayList<Integer> testBillIndices;
    protected ArrayList<Integer> testDebateIndices;
    protected boolean[][] testVotes;

    protected Set<String> keyvoteBills;
    protected ArrayList<String> policyAgendaIssues;
    protected HashMap<String, String> voteToBillMapping;
    protected HashMap<String, Integer> teapartyCaucusMapping;
    protected HashMap<String, Integer> fwEndorsementMapping;
    protected HashMap<String, Integer> tpExpressMapping;
    protected HashMap<String, Integer> spEndorsementMapping;

    protected TextDataset voteDataset;

    public VotePredExpt() {
        congressYearMap = new HashMap<String, int[]>();
        congressYearMap.put("112", new int[]{2011, 2012});
        congressYearMap.put("111", new int[]{2009, 2010});
        congressYearMap.put("110", new int[]{2007, 2008});
        congressYearMap.put("109", new int[]{2005, 2006});
    }

    public String getConfiguredExptFolder() {
        return "all";
    }

    protected void getKeyvoteBills() {
        this.keyvoteBills = new HashSet<>();
        for (String vote : debateVoteData.getVoteVocab()) {
            String billId = convertBillId(debateVoteData.getVoteProperty(vote, "bill").split(";")[0]);
            this.keyvoteBills.add(billId);
        }
    }

    protected String convertBillId(String billId) {
        if (billId.contains("H.R. ")) {
            billId = billId.replace("H.R. ", "h-");
        } else if (billId.contains("H.J.Res. ")) {
            billId = billId.replace("H.J.Res. ", "hj-");
        } else if (billId.contains("H.C.Res. ")) {
            billId = billId.replace("H.C.Res. ", "hc-");
        } else if (billId.contains("H.Res. ")) {
            billId = billId.replace("H.Res. ", "hr-");
        } else if (billId.contains("S. ")) {
            billId = billId.replace("S. ", "s-");
        } else if (billId.contains("S.C.Res. ")) {
            billId = billId.replace("S.C.Res. ", "sc-");
        } else {
            throw new RuntimeException("BillID = " + billId);
        }
        return billId;
    }

    public void adhocProcess() {
        if (verbose) {
            logln("Ad hoc processing ...");
        }

        loadFormattedData();

        File billIdFile = new File(processedDataFolder, congressNum + ".bills");
        if (verbose) {
            logln("--- Converting bill Ids to " + billIdFile);
        }
        try {
            BufferedWriter writer = IOUtils.getBufferedWriter(billIdFile);
            writer.write("ID\tBillID\n");
            for (String vote : debateVoteData.getVoteVocab()) {
                String oriBillId = debateVoteData.getVoteProperty(vote, "bill").split(";")[0];
                writer.write(vote + "\t" + convertBillId(oriBillId) + "\n");
            }
            writer.close();
        } catch (IOException e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while outputing to " + billIdFile);
        }

    }

    public void debug() {
        if (verbose) {
            logln("Debugging ...");
        }

        loadFormattedData();

        ArrayList<String> aVoc = debateVoteData.getAuthorVocab();
        ArrayList<String> bVoc = debateVoteData.getVoteVocab();
        int[][] vs = debateVoteData.getVotes();

        int aa = 0;
        String aid = aVoc.get(aa);

        System.out.println("aa = " + aa
                + ". aid = " + aid
                + ". " + debateVoteData.getAuthorProperty(aid, GTLegislator.NAME)
                + ". " + debateVoteData.getAuthorProperty(aid, GTLegislator.FW_ID));
        for (int bb = 0; bb < bVoc.size(); bb++) {
            System.out.println(">>> " + vs[aa][bb]
                    + "\t" + bVoc.get(bb)
                    + "\t" + debateVoteData.getVoteProperty(bVoc.get(bb), FWBill.TITLE));
        }
    }

    @Override
    public void setup() {
        if (verbose) {
            logln("Setting up ...");
        }

        congressNum = CLIUtils.getStringArgument(cmd, "congress-num", "112");
        datasetFolder = CLIUtils.getStringArgument(cmd, "data-folder",
                "/fs/clip-political/vietan/data/govtrack");
        numTopWords = CLIUtils.getIntegerArgument(cmd, "num-top-words", 15);
        numFolds = CLIUtils.getIntegerArgument(cmd, "num-folds", 5);
        trToDevRatio = CLIUtils.getDoubleArgument(cmd, "tr2dev-ratio", 1.0);
        teRatio = CLIUtils.getDoubleArgument(cmd, "te-ratio", 0.2);

        processedDataFolder = cmd.getOptionValue("processed-data-folder");
        experimentPath = CLIUtils.getStringArgument(cmd, "expt-folder", "vote-experiments");
        modelFolder = CLIUtils.getStringArgument(cmd, "model-folder", "model");
    }

    @Override
    public void preprocess() {
        if (verbose) {
            logln("Preprocessing ...");
        }
        try {
            loadFreedomWorksData();
            preprocessCongressionalData();
            adhocProcess();
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while preprocessing");
        }
    }

    public void preprocessVoteText() throws Exception {
        if (verbose) {
            logln("Preprocessing vote text ...");
        }
        loadFreedomWorksData();

        // vote vocab
        ArrayList<String> voteVocab = new ArrayList<String>();
        for (String v : keyvotes.keySet()) {
            voteVocab.add(v);

        }
        Collections.sort(voteVocab);

        // vote text
        ArrayList<String> voteText = new ArrayList<>();
        for (String v : voteVocab) {
            Vote keyvote = keyvotes.get(v);
            voteText.add(keyvote.getProperty(FWBill.TITLE) + " " + keyvote.getProperty(FWBill.SUMMARY));
        }

        CorpusProcessor corpProc = TextDataset.createCorpusProcessor();
        voteDataset = new TextDataset(congressNum, processedDataFolder, corpProc);
        voteDataset.setTextData(voteVocab, voteText);
        voteDataset.format(processedDataFolder);
    }

    /**
     * Load FreedomWorks data.
     */
    private void loadFreedomWorksData() throws Exception {
        if (verbose) {
            logln("--- Loading FreedomWorks data ...");
        }
        String fwFolder = cmd.getOptionValue("fw-folder");
        this.keyvotes = new HashMap<String, Vote>();
        this.voteMap = new HashMap<Integer, HashMap<String, FWVote>>();
        int[] years = congressYearMap.get(congressNum);
        this.fwYears = new FWYear[years.length];
        for (int ii = 0; ii < years.length; ii++) {
            if (verbose) {
                logln("--- --- Loading year " + years[ii]);
            }

            fwYears[ii] = new FWYear(years[ii]);

            // load legislators
            File scoreFile = new File(new File(fwFolder, congressNum),
                    years[ii] + "-house-" + FWYear.SCORE_FILE);
            fwYears[ii].inputLegislators(scoreFile);

            // load key votes
            File keyVoteFile = new File(new File(fwFolder, congressNum),
                    years[ii] + "-house-" + FWYear.KEYVOTE_FILE);
            fwYears[ii].inputKeyVotes(keyVoteFile);

            for (int rcNum : fwYears[ii].getKeyVotes().keySet()) {
                FWBill fwVote = fwYears[ii].getKeyVote(rcNum);
                Vote keyvote = new Vote(years[ii] + "_" + rcNum);
                keyvote.addProperty(FWBill.FW_VOTE_PREFERRED,
                        fwVote.getProperty(FWBill.FW_VOTE_PREFERRED));
                keyvote.addProperty(FWBill.TITLE, fwVote.getProperty(FWBill.TITLE));
                keyvote.addProperty(FWBill.ROLL_CALL, fwVote.getProperty(FWBill.ROLL_CALL));
                keyvote.addProperty(FWBill.BILL, fwVote.getProperty(FWBill.BILL));
                keyvote.addProperty(FWBill.SUMMARY, fwVote.getProperty(FWBill.SUMMARY));
                this.keyvotes.put(keyvote.getId(), keyvote);
            }

            // load votes
            File voteFile = new File(new File(fwFolder, congressNum),
                    years[ii] + "-house-" + FWYear.VOTE_FILE);
            fwYears[ii].inputVotes(voteFile);
            for (int lid : fwYears[ii].getVotes().keySet()) {
                HashMap<String, FWVote> legVotes = voteMap.get(lid);
                if (legVotes == null) {
                    legVotes = new HashMap<String, FWVote>();
                }
                ArrayList<FWVote> lv = fwYears[ii].getVotes(lid);
                for (int jj = 0; jj < lv.size(); jj++) {
                    FWVote v = lv.get(jj);
                    int rollcall = Integer.parseInt(v.getBill().getProperty(FWBill.ROLL_CALL));
                    legVotes.put(years[ii] + "_" + rollcall, v);
                }
                voteMap.put(lid, legVotes);
            }
            if (verbose) {
                logln("--- --- Loaded.");
                logln("--- --- # legislators: " + fwYears[ii].getLegislators().size());
                logln("--- --- # key votes: " + fwYears[ii].getKeyVotes().size());
            }
        }
    }

    private void loadFreedomWorksDataAdhoc() throws Exception {
        if (verbose) {
            logln("--- Loading FreedomWorks data ...");
        }
        loadFormattedData();
        HashMap<String, String> fwID2ICID = new HashMap<>();
        HashMap<String, String> fwID2ID = new HashMap<>();
        for (String a : debateVoteData.getAuthorVocab()) {
            Author author = debateVoteData.getAuthorTable().get(a);
            String icpsrid = author.getProperty(GTLegislator.ICPSRID);
            String fwid = author.getProperty(GTLegislator.FW_ID);
            fwID2ICID.put(fwid, icpsrid);
            fwID2ID.put(fwid, a);

        }

        BufferedWriter writer = IOUtils.getBufferedWriter("/fs/clip-political/vietan/herbal/budget_control.txt");

        String fwFolder = cmd.getOptionValue("fw-folder");
        this.keyvotes = new HashMap<String, Vote>();
        this.voteMap = new HashMap<Integer, HashMap<String, FWVote>>();
        int[] years = congressYearMap.get(congressNum);
        this.fwYears = new FWYear[years.length];
        for (int ii = 0; ii < years.length; ii++) {
            if (verbose) {
                logln("--- --- Loading year " + years[ii]);
            }

            fwYears[ii] = new FWYear(years[ii]);

            // load legislators
            File scoreFile = new File(new File(fwFolder, congressNum),
                    years[ii] + "-house-" + FWYear.SCORE_FILE);
            fwYears[ii].inputLegislators(scoreFile);

            // load key votes
            File keyVoteFile = new File(new File(fwFolder, congressNum),
                    years[ii] + "-house-" + FWYear.KEYVOTE_FILE);
            fwYears[ii].inputKeyVotes(keyVoteFile);

            for (int rcNum : fwYears[ii].getKeyVotes().keySet()) {
                FWBill fwVote = fwYears[ii].getKeyVote(rcNum);
                Vote keyvote = new Vote(years[ii] + "_" + rcNum);
                keyvote.addProperty(FWBill.FW_VOTE_PREFERRED,
                        fwVote.getProperty(FWBill.FW_VOTE_PREFERRED));
                keyvote.addProperty(FWBill.TITLE, fwVote.getProperty(FWBill.TITLE));
                keyvote.addProperty(FWBill.ROLL_CALL, fwVote.getProperty(FWBill.ROLL_CALL));
                keyvote.addProperty(FWBill.BILL, fwVote.getProperty(FWBill.BILL));
                keyvote.addProperty(FWBill.SUMMARY, fwVote.getProperty(FWBill.SUMMARY));
                this.keyvotes.put(keyvote.getId(), keyvote);
            }

            // load votes
            File voteFile = new File(new File(fwFolder, congressNum),
                    years[ii] + "-house-" + FWYear.VOTE_FILE);
            fwYears[ii].inputVotes(voteFile);

            int tpWith = 0;
            int ntpWith = 0;
            int tpAgainst = 0;
            int ntpAgainst = 0;

            for (int lid : fwYears[ii].getVotes().keySet()) {
                HashMap<String, FWVote> legVotes = voteMap.get(lid);
                if (legVotes == null) {
                    legVotes = new HashMap<String, FWVote>();
                }
                ArrayList<FWVote> lv = fwYears[ii].getVotes(lid);
                for (int jj = 0; jj < lv.size(); jj++) {
                    int rollcall = fwYears[ii].getKeyRollCalls().get(jj);
                    legVotes.put(years[ii] + "_" + rollcall, lv.get(jj));
                }
                voteMap.put(lid, legVotes);

                // debug
//                System.out.println("lid = " + lid);
                for (String vv : legVotes.keySet()) {
                    FWVote fwvote = legVotes.get(vv);

                    String title = fwvote.getBill().getProperty(FWBill.TITLE);

//                    if (!title.contains("Budget Control Act")) {
//                    if (!title.contains("replace the Paul Ryan budget")) {
//                    if (!title.contains("amendment containing the Republican")) {
                    if (!title.contains("Taxpayer Relief Act")) {
                        continue;
                    }

                    FWVote.VoteType vote = fwvote.getType();
                    String name = fwvote.getLegislator().getProperty(GTLegislator.NAME);
                    String icpsrID = fwID2ICID.get(Integer.toString(lid));
                    String voteStr;
                    if (vote == FWVote.VoteType.AGAINST) {
                        voteStr = "0";
                    } else if (vote == FWVote.VoteType.WITH) {
                        voteStr = "1";
                    } else {
                        continue;
                    }
                    String aid = fwID2ID.get(Integer.toString(lid));
                    Author aaa = debateVoteData.getAuthorTable().get(aid);
                    if (aaa == null) {
                        continue;
                    }
                    if (aaa.getProperty(GTLegislator.PARTY).equals("Democrat")) {
                        continue;
                    }

                    String str = lid
                            + "\t" + icpsrID
                            + "\t" + aaa.getId()
                            + "\t" + name
                            + "\t" + aaa.getProperty(GTLegislator.NAME)
                            + "\t" + aaa.getProperty(GTLegislator.PARTY)
                            + "\t" + getTeaPartyCaucus(icpsrID)
                            + "\t" + voteStr
                            + "\t" + title;
                    writer.write(str + "\n");
                    System.out.println(str);

                    int tp = getTeaPartyCaucus(icpsrID);
                    int v = Integer.parseInt(voteStr);
                    if (tp == 1 && v == 1) {
                        tpWith++;
                    } else if (tp == 1 && v == 0) {
                        tpAgainst++;
                    } else if (tp == 0 && v == 1) {
                        ntpWith++;
                    } else {
                        ntpAgainst++;
                    }
                }
            }
            if (verbose) {
                logln("--- --- Loaded.");
                logln("--- --- # legislators: " + fwYears[ii].getLegislators().size());
                logln("--- --- # key votes: " + fwYears[ii].getKeyVotes().size());
            }

            System.out.println("TP-W = " + tpWith);
            System.out.println("TP-A = " + tpAgainst);
            System.out.println("NTP-W = " + ntpWith);
            System.out.println("NTP-A = " + ntpAgainst);

//            break; // for 111
        }
        writer.close();

        System.exit(1);
    }

    /**
     * Process text data.
     */
    private void preprocessCongressionalData() throws Exception {
        if (verbose) {
            logln("--- Loading Congressional data ...");
        }
        File congressFolder = new File(new File(datasetFolder, congressNum), "processedV5");
        data = Congress.loadProcessedCongress(congressNum,
                datasetFolder,
                new File(congressFolder, "bills").getAbsolutePath(),
                new File(congressFolder, "debates").getAbsolutePath(),
                new File(congressFolder, "legislators.txt").getAbsolutePath(),
                "/fs/clip-political/vietan/data/govtrack/addinfo/112th-House-Republicans.txt",
                null);
        data.setCongressYear(fwYears);

        // load FWScore with FW_ID
        Congress.getFreedomWorksScore(data.getLegislators(), fwYears[0], PRE_SCORE);
        Congress.getFreedomWorksScore(data.getLegislators(), fwYears[1], POS_SCORE);

        if (verbose) {
            logln("--- --- Loaded.");
            logln("--- Processing text data ...");
        }
        // vote vocab
        ArrayList<String> voteVocab = new ArrayList<String>();
        for (String v : keyvotes.keySet()) {
            voteVocab.add(v);

        }
        Collections.sort(voteVocab);

        // vote text
        ArrayList<String> voteText = new ArrayList<>();
        for (String v : voteVocab) {
            Vote keyvote = keyvotes.get(v);
            String keyvoteTitle = keyvote.getProperty(FWBill.TITLE);
            String keyvoteSummary = keyvote.getProperty(FWBill.SUMMARY);
            voteText.add(keyvoteTitle + " " + keyvoteSummary);
        }
        // actual votes
        HashMap<String, HashMap<Integer, Integer>> rawVotes
                = new HashMap<String, HashMap<Integer, Integer>>();
        ArrayList<String> authorVocab = new ArrayList<String>();
        for (String legid : data.getLegislators().keySet()) {
            GTLegislator leg = data.getLegislator(legid);
            if (leg.getType().equals("sen")) {
                continue;
            }

            if (leg.getProperty(GTLegislator.FW_ID) == null) {
                logln("Skipping legislator with no FW_ID: " + leg.toString());
                continue;
            }
            int legFWId = Integer.parseInt(leg.getProperty(GTLegislator.FW_ID));

            HashMap<String, FWVote> vs = this.voteMap.get(legFWId);
            if (vs == null) {
                logln("Skipping legislator with no vote: " + leg.toString());
                continue;
            }

            HashMap<Integer, Integer> modvs = new HashMap<Integer, Integer>();
            for (String voteId : vs.keySet()) {
                int vidx = voteVocab.indexOf(voteId);
                FWVote v = vs.get(voteId);
                if (v.getType() == FWVote.VoteType.WITH) {
                    modvs.put(vidx, WITH);
                } else if (v.getType() == FWVote.VoteType.AGAINST) {
                    modvs.put(vidx, AGAINST);
                }
            }
            rawVotes.put(legid, modvs);
            authorVocab.add(legid);
        }

        ArrayList<String> voteProperties = new ArrayList<String>();
        voteProperties.add(FWBill.ROLL_CALL);
        voteProperties.add(FWBill.BILL);
        voteProperties.add(FWBill.FW_VOTE_PREFERRED);
        voteProperties.add(FWBill.TITLE);
        voteProperties.add(FWBill.SUMMARY);

        // properties of legislators
        ArrayList<String> legislatorProperties = new ArrayList<String>();
        legislatorProperties.add(GTLegislator.ICPSRID);
        legislatorProperties.add(GTLegislator.FW_ID);
        legislatorProperties.add(GTLegislator.NAME);
        legislatorProperties.add(GTLegislator.PARTY);
        legislatorProperties.add(GTLegislator.TYPE);
        legislatorProperties.add(GTLegislator.NOMINATE_SCORE1);
        legislatorProperties.add(GTLegislator.FW_SCORE);
        legislatorProperties.add(GTLegislator.TP_Caucus);

        CorpusProcessor corpProc = TextDataset.createCorpusProcessor();
        debateVoteData = new AuthorVoteTextDataset(data.getName(), processedDataFolder, corpProc);
        debateVoteData.setVoteVocab(voteVocab);
        debateVoteData.setAuthorVocab(authorVocab);
        debateVoteData.setAuthorVotes(rawVotes);

        Debate debate = data.getDebateData();
        debateVoteData.setTextData(debate.getDocIdList(), debate.getTextList());
        debateVoteData.setVoteText(voteText);
        debateVoteData.setAuthorList(debate.getSpeakerList());
        debateVoteData.setBillList(debate.getBillList());
        debateVoteData.setAuthorPropertyNames(legislatorProperties);
        debateVoteData.setVotePropertyNames(voteProperties);
        for (String author : authorVocab) {
            setLegislatorPropertyValues(debateVoteData, author);
        }
        for (String vid : keyvotes.keySet()) {
            setVotePropertyValues(debateVoteData, vid);
        }
        debateVoteData.setHasSentences(true);
        debateVoteData.format(processedDataFolder);

        // process bill data
        billData = data.getBillData();
        billData.setCorpusProcessor(debateVoteData.getCorpusProcessor());
        billData.setHasSentences(true);
        billData.format(processedDataFolder);
    }

    private void setVotePropertyValues(AuthorVoteTextDataset phaseData, String vid) {
        Vote vote = keyvotes.get(vid);
        phaseData.setVoteProperty(vid, FWBill.BILL, vote.getProperty(FWBill.BILL));
        phaseData.setVoteProperty(vid, FWBill.TITLE, vote.getProperty(FWBill.TITLE));
        phaseData.setVoteProperty(vid, FWBill.FW_VOTE_PREFERRED,
                vote.getProperty(FWBill.FW_VOTE_PREFERRED));
        phaseData.setVoteProperty(vid, FWBill.ROLL_CALL, vote.getProperty(FWBill.ROLL_CALL));
    }

    private void setLegislatorPropertyValues(AuthorVoteTextDataset phaseData, String lid) {
        GTLegislator leg = data.getLegislator(lid);
        phaseData.setAuthorProperty(lid, GTLegislator.ICPSRID, leg.getProperty(GTLegislator.ICPSRID));
        phaseData.setAuthorProperty(lid, GTLegislator.NAME, leg.getName());
        phaseData.setAuthorProperty(lid, GTLegislator.PARTY, leg.getParty());
        phaseData.setAuthorProperty(lid, "type", leg.getType());
        phaseData.setAuthorProperty(lid, GTLegislator.FW_ID, leg.getProperty(GTLegislator.FW_ID));
        phaseData.setAuthorProperty(lid, GTLegislator.NOMINATE_SCORE1,
                leg.getProperty(GTLegislator.NOMINATE_SCORE1));
        phaseData.setAuthorProperty(lid, GTLegislator.FW_SCORE,
                leg.getProperty(GTLegislator.FW_SCORE));
        phaseData.setAuthorProperty(lid, GTLegislator.TP_Caucus,
                leg.getProperty(GTLegislator.TP_Caucus));
    }

    protected void loadFormattedData() {
        if (verbose) {
            logln("--- Loading debate data from " + processedDataFolder);
        }
        debateVoteData = new AuthorVoteTextDataset(congressNum, processedDataFolder);
        debateVoteData.loadFormattedData(processedDataFolder);
        votes = debateVoteData.getVotes();
        debateVoteData.prepareTopicCoherence(numTopWords);

        if (verbose) {
            logln("--- Loading bill data from " + processedDataFolder);
        }
        billData = new Bill("bill-" + congressNum, processedDataFolder);
        billData.loadFormattedData(processedDataFolder);

        if (verbose) {
            logln("--- Loading vote text from " + processedDataFolder);
        }
        voteDataset = new TextDataset(congressNum, processedDataFolder);
        voteDataset.loadFormattedData(processedDataFolder);

        policyAgendaIssues = new ArrayList<>();
        for (String issue : billData.getTopicVocab()) {
            if (!issue.equals("Other")) {
                policyAgendaIssues.add(issue);
            }
        }

        File billIdFile = new File(processedDataFolder, congressNum + ".bills");
        if (billIdFile.exists()) {
            this.voteToBillMapping = loadVote2BillMapping(billIdFile);
        }

        if (verbose) {
            logln("--- Data loaded.");
            logln("--- Policy Agendas Topics:");
            for (String issue : policyAgendaIssues) {
                logln("--- --- " + issue);
            }
        }

        loadTeaPartyMapping();
    }

    private int getTeaPartyCaucus(String icpsrID) {
        Integer tpc = this.teapartyCaucusMapping.get(icpsrID);
        if (tpc == null) {
            return 0;
        } else {
            return tpc;
        }
    }

    private void loadTeaPartyMapping() {
        this.teapartyCaucusMapping = new HashMap<>();
        this.fwEndorsementMapping = new HashMap<>();
        this.tpExpressMapping = new HashMap<>();
        this.spEndorsementMapping = new HashMap<>();
        try {
            String file = "/fs/clip-political/"
                    + "vietan/data/govtrack/addinfo/112th-House-Republicans.txt";
            System.out.println("Loading tea party mapping from " + file);
            BufferedReader reader = IOUtils.getBufferedReader(file);
            reader.readLine();
            String line;
            while ((line = reader.readLine()) != null) {
                String[] sline = line.split("\t");
                if (sline.length == 0) {
                    break;
                }
                System.out.println("line = " + line);
                String icpsrId = sline[0];
                this.teapartyCaucusMapping.put(icpsrId, Integer.parseInt(sline[7]));
                this.fwEndorsementMapping.put(icpsrId, sline[8].isEmpty() ? 0 : Integer.parseInt(sline[8]));
                this.tpExpressMapping.put(icpsrId, Integer.parseInt(sline[9]));
                this.spEndorsementMapping.put(icpsrId, Integer.parseInt(sline[10]));
            }
            reader.close();
            System.out.println("# TP Caucus: " + this.teapartyCaucusMapping.size());
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException();
        }
    }

    private HashMap<String, String> loadVote2BillMapping(File file) {
        if (verbose) {
            logln("--- Loading vote to bill mapping from " + file);
        }
        HashMap<String, String> vote2Bills = new HashMap<>();
        try {
            BufferedReader reader = IOUtils.getBufferedReader(file);
            reader.readLine();
            String line;
            while ((line = reader.readLine()) != null) {
                String[] sline = line.split("\t");
                vote2Bills.put(sline[0], sline[1]);
            }
            reader.close();

            if (verbose) {
                logln("--- --- Loaded vote-to-bill mapping: " + vote2Bills.size());
            }
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException();
        }
        return vote2Bills;
    }

    /**
     * Estimate on all data.
     */
    @Override
    public void run() {
        if (verbose) {
            logln("Running models ...");
        }

        setupSampling();

        loadFormattedData();

        trainAuthorIndices = new ArrayList<>();
        for (int aa = 0; aa < debateVoteData.getAuthorVocab().size(); aa++) {
            trainAuthorIndices.add(aa);
        }

        trainVotes = new boolean[votes.length][];
        for (int aa = 0; aa < votes.length; aa++) {
            trainVotes[aa] = new boolean[votes[aa].length];
            for (int bb = 0; bb < votes[aa].length; bb++) {
                if (votes[aa][bb] != Vote.MISSING) {
                    trainVotes[aa][bb] = true;
                }
            }
        }

        trainDebateIndices = new ArrayList<>();
        for (int dd = 0; dd < debateVoteData.getWords().length; dd++) {
            int author = debateVoteData.getAuthors()[dd];
            if (trainAuthorIndices.contains(author)) {
                this.trainDebateIndices.add(dd);
            }
        }

        if (verbose) {
            logln("--- # lawmakers: " + trainAuthorIndices.size()
                    + " / " + debateVoteData.getAuthorVocab().size());
            logln("--- # debates: " + this.trainDebateIndices.size()
                    + " / " + debateVoteData.getWords().length);
        }

        File configureFolder = new File(new File(experimentPath, congressNum),
                getConfiguredExptFolder());
        IOUtils.createFolder(configureFolder);

        this.runModel(configureFolder);
    }

    protected int[] concatArray(int[] a, int[] b) {
        int[] ab = new int[a.length + b.length];
        System.arraycopy(a, 0, ab, 0, a.length);
        System.arraycopy(b, 0, ab, a.length, b.length);
        return ab;
    }

    /**
     * Run a model.
     *
     * @param outputFolder Output folder
     */
    protected void runModel(File outputFolder) {
        int[][] voteTextWords = voteDataset.getWords();

        // get words of training bills
        trainVoteWords = new int[debateVoteData.getVoteVocab().size()][];
        trainVoteTopics = new int[debateVoteData.getVoteVocab().size()];

        List<String> billIdList = Arrays.asList(billData.getDocIds());
        for (int bb = 0; bb < trainVoteWords.length; bb++) {
            String keyvote = debateVoteData.getVoteVocab().get(bb);
            String billId = voteToBillMapping.get(keyvote);
            int idx = billIdList.indexOf(billId);

            trainVoteWords[bb] = concatArray(billData.getWords()[idx], voteTextWords[bb]);
            trainVoteTopics[bb] = billData.getTopics()[idx];
        }

        String model = CLIUtils.getStringArgument(cmd, "model", "random");
        switch (model) {
            case "random":
                runRandom(outputFolder);
                break;
            case "lda-bill":
                runLDABills(outputFolder);
                break;
            case "lda-debate":
                runLDADebates(outputFolder);
                break;
            case "ideal-point":
                runIdealPoint(outputFolder);
                break;
            case "bayesian-ideal-point":
                runBayesianIdealPoint(outputFolder);
                break;
            case "bayesian-mult-ideal-point":
                runBayesianMultIdealPoint(outputFolder);
                break;
            case "bayesian-mult-ideal-point-owlqn":
                runBayesianMultIdealPointOWLQN(outputFolder);
                break;
            case "lexical-ideal-point":
                runLexicalIdealPoint(outputFolder);
                break;
            case "slda-ideal-point":
                runSLDAIdealPoint(outputFolder);
                break;
            case "recursive-slda-ideal-point":
                runRecursiveSLDAIdealPoint(outputFolder);
                break;
            case "hier-mult-tipm":
                runHierMultTIPM(outputFolder);
                break;
            case "hier-mult-shdp":
                runHierMultSHDP(outputFolder);
                break;
            case "hier-single-tipm":
                runHierSingleTIPM(outputFolder);
                break;
            case "lexical-slda-ideal-point":
                runLexicalSLDAIdealPoint(outputFolder);
                break;
            case "hybrid-slda-ideal-point":
                runHybridSLDAIdealPoint(outputFolder);
                break;
            case "hybrid-slda-multiple-ideal-point":
                runHybridSLDAMultipleIdealPoint(outputFolder);
                break;
            case "slda-mult-ideal-point":
                runSLDAMultIdealPoint(outputFolder);
                break;
            case "snlda-ideal-point":
                runSNLDAIdealPoint(outputFolder);
                break;
            case "lexical-snlda-ideal-point":
                runLexicalSNLDAIdealPoint(outputFolder);
                break;
            case "hybrid-snlda-ideal-point":
                runHybridSNLDAIdealPoint(outputFolder);
                break;
            case "snlda-mult-ideal-point":
                runSNLDAMultIdealPoint(outputFolder);
                break;
            case "snhdp-ideal-point":
                runSNHDPIdealPoint(outputFolder);
                break;
            case "hybrid-snhdp-ideal-point":
                runHybridSNHDPIdealPoint(outputFolder);
                break;
            case "summarize":
                summarizeAuthorScores(outputFolder);
                break;
            case "none":
                logln("Doing nothing :D");
                break;
            default:
                throw new RuntimeException("Model " + model + " not supported");
        }
    }

    /**
     * Summarize author scores estimated by different methods.
     *
     * @param estimateFolder
     */
    private void summarizeAuthorScores(File estimateFolder) {
        if (verbose) {
            logln("--- Summarizing author scores ...");
        }
        MinMaxNormalizer mmNorm;
        int A = this.trainAuthorIndices.size();
        Author[] authors = new Author[A];
        ArrayList<String> headers = new ArrayList<>();

        // basic info
        headers.add(GTLegislator.FW_ID);
        headers.add(GTLegislator.NAME);
        headers.add(GTLegislator.PARTY);
        headers.add("NumWithVotes");
        headers.add("NumAgainstVotes");
        headers.add("FWScore");
        for (int ii = 0; ii < A; ii++) {
            int aa = this.trainAuthorIndices.get(ii);
            String authorId = debateVoteData.getAuthorVocab().get(aa);
            authors[ii] = new Author(authorId);
            authors[ii].addProperty(GTLegislator.FW_ID, debateVoteData.getAuthorProperty(
                    authorId, GTLegislator.FW_ID));
            authors[ii].addProperty(GTLegislator.NAME, debateVoteData.getAuthorProperty(
                    authorId, GTLegislator.NAME));
            authors[ii].addProperty(GTLegislator.PARTY, debateVoteData.getAuthorProperty(
                    authorId, GTLegislator.PARTY));

            int withCount = 0;
            int againstCount = 0;
            for (int bb = 0; bb < votes[aa].length; bb++) {
                if (trainVotes[aa][bb]) {
                    if (votes[aa][bb] == Vote.WITH) {
                        withCount++;
                    } else if (votes[aa][bb] == Vote.AGAINST) {
                        againstCount++;
                    }
                }
            }
            double ratio = (double) withCount / (withCount + againstCount);
            authors[ii].addProperty("NumWithVotes", Integer.toString(withCount));
            authors[ii].addProperty("NumAgainstVotes", Integer.toString(againstCount));
            authors[ii].addProperty("FWScore", Double.toString(ratio));
        }

        // ideal point
        IdealPoint ip = new IdealPoint("ideal-point");
        ip.configure(1, 0.01, 10000);
        ip.setAuthorVocab(debateVoteData.getAuthorVocab());
        ip.setVoteVocab(debateVoteData.getVoteVocab());
        File ipFolder = new File(estimateFolder, ip.getName());
        ip.input(new File(ipFolder, MODEL_FILE));
        mmNorm = new MinMaxNormalizer(ip.getUs(), 0.0, 1.0);
        for (int ii = 0; ii < A; ii++) {
            authors[ii].addProperty(ip.getBasename(), Double.toString(mmNorm.normalize(ip.getUs()[ii])));
        }
        headers.add(ip.getBasename());

        // bayesian ideal point
        BayesianIdealPoint bip = new BayesianIdealPoint("bayesian-ideal-point");
        bip.configure(1, 0.01, 10000, 0, 2.5);
        bip.setAuthorVocab(debateVoteData.getAuthorVocab());
        bip.setVoteVocab(debateVoteData.getVoteVocab());
        File bipFolder = new File(estimateFolder, bip.getName());
        bip.input(new File(bipFolder, MODEL_FILE));
        mmNorm = new MinMaxNormalizer(bip.getUs(), 0.0, 1.0);
        for (int ii = 0; ii < A; ii++) {
            authors[ii].addProperty(bip.getBasename(), Double.toString(mmNorm.normalize(bip.getUs()[ii])));
        }
        headers.add(bip.getBasename());

        // Author-SLDA ideal point
        SLDAIdealPoint sampler = new SLDAIdealPoint();
        sampler.setWordVocab(debateVoteData.getWordVocab());
        sampler.setAuthorVocab(debateVoteData.getAuthorVocab());
        sampler.setVoteVocab(debateVoteData.getVoteVocab());
        sampler.configure(estimateFolder.getAbsolutePath(),
                debateVoteData.getWordVocab().size(), 20,
                0.1, 0.1, 1.0, 0.0, 2.5, 1.0, 0.01,
                initState, paramOpt,
                10, 25, 3, 1);
        sampler.setupData(trainDebateIndices,
                debateVoteData.getWords(),
                debateVoteData.getAuthors(),
                votes,
                trainAuthorIndices,
                trainBillIndices,
                trainVotes);
        sampler.inputFinalState();
        mmNorm = new MinMaxNormalizer(sampler.getUs(), 0.0, 1.0);

        double[][] authorTopicProps = sampler.getAuthorTopicProportions();
        for (String label : billData.getTopicVocab()) {
            headers.add(label);
        }

        for (int ii = 0; ii < A; ii++) {
            authors[ii].addProperty(sampler.getBasename(),
                    Double.toString(mmNorm.normalize(sampler.getUs()[ii])));
            for (int ll = 0; ll < billData.getTopicVocab().size(); ll++) {
                String label = billData.getTopicVocab().get(ll);
                authors[ii].addProperty(label, Double.toString(authorTopicProps[ii][ll]));
            }
        }
        headers.add(sampler.getBasename());

        try {
            File authorEstimateFile = new File(estimateFolder, "authors.score");
            if (verbose) {
                logln("--- --- Summarizing to " + authorEstimateFile);
            }

            // header
            BufferedWriter writer = IOUtils.getBufferedWriter(authorEstimateFile);
            writer.write("ID");
            for (String prop : headers) {
                writer.write("\t" + prop);
            }
            writer.write("\n");

            // authors
            for (Author author : authors) {
                writer.write(author.getId());
                for (String prop : headers) {
                    writer.write("\t" + author.getProperty(prop));
                }
                writer.write("\n");
            }

            writer.close();
        } catch (IOException e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while summarizing author scores");
        }
    }

    protected void runRandom(File outputFolder) {
        if (verbose) {
            logln("--- --- Running random predictor ...");
        }
        RandomPredictor randPred = new RandomPredictor("random");
        File predFolder = new File(outputFolder, randPred.getName());

        if (cmd.hasOption("testvote") || cmd.hasOption("testauthor")) {
            SparseVector[] predictions = randPred.test(testVotes);
            File teResultFolder = new File(predFolder, TEST_PREFIX + RESULT_FOLDER);
            IOUtils.createFolder(teResultFolder);
            AbstractModel.outputPerformances(new File(teResultFolder, RESULT_FILE),
                    AbstractVotePredictor.evaluateAll(votes, testVotes, predictions));
        }
    }

    protected void runLDADebates(File outputFolder) {
        double alpha = CLIUtils.getDoubleArgument(cmd, "alpha", 0.1);
        double beta = CLIUtils.getDoubleArgument(cmd, "beta", 0.1);
        int V = debateVoteData.getWordVocab().size();

        int K;
        double[][] issuePhis;
        if (cmd.hasOption("K")) {
            issuePhis = null;
            K = Integer.parseInt(cmd.getOptionValue("K"));
        } else {
            issuePhis = estimateIssues();
            K = issuePhis.length;
        }

        LDA sampler = new LDA();
        sampler.setVerbose(cmd.hasOption("v"));
        sampler.setDebug(cmd.hasOption("d"));
        sampler.setLog(true);
        sampler.setReport(true);
        sampler.setWordVocab(debateVoteData.getWordVocab());

        sampler.configure(outputFolder.getAbsolutePath(), V, K,
                alpha, beta,
                initState, paramOpt,
                burn_in, max_iters, sample_lag, report_interval);
        File samplerFolder = new File(sampler.getSamplerFolderPath());
        IOUtils.createFolder(samplerFolder);

        if (cmd.hasOption("train")) {
            sampler.train(debateVoteData.getWords(), null);
            sampler.initialize(null, issuePhis);
            sampler.iterate();
            sampler.outputTopicTopWords(new File(samplerFolder, TopWordFile), numTopWords);
        }
    }

    protected void runLDABills(File estimateFolder) {
        Initializer init = new Initializer(billData.getWords(),
                billData.getTopics(), debateVoteData.getWords(),
                billData.getTopicVocab().size(), billData.getWordVocab().size());

        File priorFile = new File(estimateFolder, "priors.txt");
        if (!priorFile.exists()) {
            init.estimate();
            IOUtils.output2DArray(priorFile, init.labelPriors);
        }
        double[][] labelPriors = IOUtils.input2DArray(priorFile);
        int K = billData.getTopicVocab().size();
        int V = debateVoteData.getWordVocab().size();

        double[][] priors = new double[K + 1][];
        for (int kk = 0; kk < K; kk++) {
            priors[kk] = new double[V];
            System.arraycopy(labelPriors[kk], 0, priors[kk], 0, V);
        }
        priors[K] = new double[V]; // background
        for (int[] word : debateVoteData.getWords()) {
            for (int nn = 0; nn < word.length; nn++) {
                priors[K][word[nn]]++;
            }
        }
        double sum = StatUtils.sum(priors[K]);
        for (int vv = 0; vv < V; vv++) {
            priors[K][vv] /= sum;
        }

        // debug
        System.out.println("Background");
        String[] bgTopWords = MiscUtils.getTopWords(billData.getWordVocab(), priors[K], 15);
        for (String w : bgTopWords) {
            System.out.println("\t" + w);
        }

        for (int ll = 0; ll < K; ll++) {
            System.out.println("\n" + billData.getTopicVocab().get(ll));
            String[] topWords = MiscUtils.getTopWords(billData.getWordVocab(), priors[ll], 15);
            for (String w : topWords) {
                System.out.print("\t" + w);
            }
            System.out.println("\n");
        }

        double alpha = CLIUtils.getDoubleArgument(cmd, "alpha", 0.1);
        double beta = CLIUtils.getDoubleArgument(cmd, "beta", 0.1);

        LDA sampler = new LDA();
        sampler.setVerbose(cmd.hasOption("v"));
        sampler.setDebug(cmd.hasOption("d"));
        sampler.setLog(true);
        sampler.setReport(true);
        sampler.setWordVocab(debateVoteData.getWordVocab());

        sampler.configure(estimateFolder.getAbsolutePath(), V, K + 1,
                alpha, beta,
                initState, paramOpt,
                burn_in, max_iters, sample_lag, report_interval);
        File samplerFolder = new File(sampler.getSamplerFolderPath());
        IOUtils.createFolder(samplerFolder);

        if (cmd.hasOption("train")) {
            sampler.train(debateVoteData.getWords(), null);
            sampler.initialize(null, priors);
            sampler.iterate();
            sampler.outputTopicTopWords(new File(samplerFolder, TopWordFile), numTopWords);
        }
    }

    protected void runIdealPoint(File outputFolder) {
        double alpha = CLIUtils.getDoubleArgument(cmd, "alpha", 5.0);
        double eta = CLIUtils.getDoubleArgument(cmd, "eta", 0.01);

        IdealPoint pred = new IdealPoint("ideal-point");
        pred.configure(alpha, eta, max_iters);
        pred.setAuthorVocab(debateVoteData.getAuthorVocab());
        pred.setVoteVocab(debateVoteData.getVoteVocab());

        File predFolder = new File(outputFolder, pred.getName());

        if (cmd.hasOption("train")) {
            pred.setTrain(votes, trainAuthorIndices, trainBillIndices, trainVotes);
            pred.train();
            IOUtils.createFolder(predFolder);
            pred.output(new File(predFolder, MODEL_FILE));
            outputAuthorScore(new File(predFolder, AuthorScoreFile),
                    debateVoteData.getAuthorVocab(),
                    trainAuthorIndices,
                    trainVotes,
                    pred.getUs(),
                    debateVoteData.getAuthorTable());
            outputVoteScores(new File(predFolder, VoteScoreFile),
                    debateVoteData.getVoteVocab(), this.trainBillIndices,
                    pred.getXs(), pred.getYs(),
                    debateVoteData.getVoteTable());
        }

        if (cmd.hasOption("testvote")) {
            pred.input(new File(predFolder, MODEL_FILE));

            SparseVector[] predictions = pred.test(testVotes);
            File teResultFolder = new File(predFolder, TEST_PREFIX + RESULT_FOLDER);
            IOUtils.createFolder(teResultFolder);
            AbstractModel.outputPerformances(new File(teResultFolder, RESULT_FILE),
                    AbstractVotePredictor.evaluateAll(votes, testVotes, predictions));
        }
    }

    protected void runBayesianIdealPoint(File outputFolder) {
        double alpha = CLIUtils.getDoubleArgument(cmd, "alpha", 5.0);
        double eta = CLIUtils.getDoubleArgument(cmd, "eta", 0.01);
        double mu = CLIUtils.getDoubleArgument(cmd, "mu", 0.0);
        double sigma = CLIUtils.getDoubleArgument(cmd, "sigma", 2.5);

        BayesianIdealPoint pred = new BayesianIdealPoint("bayesian-ideal-point");
        pred.configure(alpha, eta, max_iters, mu, sigma);
        pred.setAuthorVocab(debateVoteData.getAuthorVocab());
        pred.setVoteVocab(debateVoteData.getVoteVocab());

        File predFolder = new File(outputFolder, pred.getName());

        if (cmd.hasOption("train")) {
            pred.setTrain(votes, trainAuthorIndices, trainBillIndices, trainVotes);
            pred.train();
            IOUtils.createFolder(predFolder);
            pred.output(new File(predFolder, MODEL_FILE));
            outputAuthorScore(new File(predFolder, AuthorScoreFile),
                    debateVoteData.getAuthorVocab(),
                    trainAuthorIndices,
                    trainVotes,
                    pred.getUs(),
                    debateVoteData.getAuthorTable());
            outputVoteScores(new File(predFolder, VoteScoreFile),
                    debateVoteData.getVoteVocab(),
                    trainBillIndices,
                    pred.getXs(), pred.getYs(),
                    debateVoteData.getVoteTable());

            // evaluate on training data
//            SparseVector[] predictions = pred.test(trainVotes);
//            File trResultFolder = new File(predFolder, TRAIN_PREFIX + RESULT_FOLDER);
//            IOUtils.createFolder(trResultFolder);
//            AbstractModel.outputPerformances(new File(trResultFolder, RESULT_FILE),
//                    AbstractVotePredictor.evaluateAll(votes, trainVotes, predictions));
        }

        if (cmd.hasOption("testvote")) {
            pred.input(new File(predFolder, MODEL_FILE));
            SparseVector[] predictions = pred.test(testVotes);
            File teResultFolder = new File(predFolder, TEST_PREFIX + RESULT_FOLDER);
            IOUtils.createFolder(teResultFolder);
            AbstractModel.outputPerformances(new File(teResultFolder, RESULT_FILE),
                    AbstractVotePredictor.evaluateAll(votes, testVotes, predictions));
        }

        if (cmd.hasOption("dev")) {
            if (verbose) {
                logln("Tuning " + pred.getBasename());
            }
            File devFolder = new File(outputFolder, pred.getBasename());
            IOUtils.createFolder(devFolder);
            try {
                BufferedWriter writer = IOUtils.getBufferedWriter(new File(devFolder, "dev.txt"));
                max_iters = 10000;
                double[] alphas = {1.0};
                double[] etas = {0.005, 0.01};
                double[] sigmas = {1.0, 2.5, 5.0};
                boolean header = true;
                for (double a : alphas) {
                    for (double e : etas) {
                        for (double s : sigmas) {
                            pred.configure(a, e, max_iters, mu, s);
                            predFolder = new File(devFolder, pred.getName());
                            IOUtils.createFolder(predFolder);

                            pred.setTrain(votes, null, null, trainVotes);
                            pred.train();
                            pred.output(new File(predFolder, MODEL_FILE));
                            outputAuthorScore(new File(predFolder, AuthorScoreFile),
                                    debateVoteData.getAuthorVocab(),
                                    trainAuthorIndices,
                                    trainVotes,
                                    pred.getUs(),
                                    debateVoteData.getAuthorTable());
                            outputVoteScores(new File(predFolder, VoteScoreFile),
                                    debateVoteData.getVoteVocab(), this.trainBillIndices,
                                    pred.getXs(), pred.getYs(),
                                    debateVoteData.getVoteTable());

                            pred.input(new File(predFolder, MODEL_FILE));
                            SparseVector[] predictions = pred.test(testVotes);
                            ArrayList<Measurement> results = AbstractVotePredictor
                                    .evaluateAll(votes, testVotes, predictions);
                            AbstractModel.outputPerformances(new File(predFolder, RESULT_FILE),
                                    results);

                            if (header) {
                                writer.write("alpha\teta\tsigma");
                                for (Measurement m : results) {
                                    writer.write("\t" + m.getName());
                                }
                                writer.write("\n");
                                header = false;
                            }

                            writer.write(a + "\t" + e + "\t" + s);
                            for (Measurement m : results) {
                                writer.write("\t" + m.getValue());
                            }
                            writer.write("\n");
                        }
                    }
                }
                writer.close();
            } catch (IOException e) {
                e.printStackTrace();
                throw new RuntimeException("Exception while tuning");
            }
        }
    }

    protected void runBayesianMultIdealPoint(File outputFolder) {
        double alpha = CLIUtils.getDoubleArgument(cmd, "alpha", 5.0);
        double eta = CLIUtils.getDoubleArgument(cmd, "eta", 0.01);
        double mu = CLIUtils.getDoubleArgument(cmd, "mu", 0.0);
        double sigma = CLIUtils.getDoubleArgument(cmd, "sigma", 2.5);
        int K = CLIUtils.getIntegerArgument(cmd, "K", 10);

        BayesianMultIdealPoint pred = new BayesianMultIdealPoint("bayesian-mult-ideal-point");
        pred.configure(alpha, eta, max_iters, mu, sigma, K);
        pred.setAuthorVocab(debateVoteData.getAuthorVocab());
        pred.setVoteVocab(debateVoteData.getVoteVocab());

        File predFolder = new File(outputFolder, pred.getName());
        if (cmd.hasOption("train")) {
            pred.setTrain(votes, trainAuthorIndices, trainBillIndices, trainVotes);
            pred.train();
            IOUtils.createFolder(predFolder);
            pred.output(new File(predFolder, MODEL_FILE));

            // evaluate on training data
            File trResultFolder = new File(predFolder, TRAIN_PREFIX + RESULT_FOLDER);
            IOUtils.createFolder(trResultFolder);
            SparseVector[] predictions = pred.test(trainVotes);
            AbstractModel.outputPerformances(new File(trResultFolder, RESULT_FILE),
                    AbstractVotePredictor.evaluateAll(votes, trainVotes, predictions));
        }

        if (cmd.hasOption("testvote")) {
            pred.input(new File(predFolder, MODEL_FILE));
            SparseVector[] predictions = pred.test(testVotes);
            File teResultFolder = new File(predFolder, TEST_PREFIX + RESULT_FOLDER);
            IOUtils.createFolder(teResultFolder);
            AbstractModel.outputPerformances(new File(teResultFolder, RESULT_FILE),
                    AbstractVotePredictor.evaluateAll(votes, testVotes, predictions));
        }
    }

    protected void runBayesianMultIdealPointOWLQN(File outputFolder) {
        double l1 = CLIUtils.getDoubleArgument(cmd, "l1", 0.0);
        double l2 = CLIUtils.getDoubleArgument(cmd, "l2", 2.5);
        int K = CLIUtils.getIntegerArgument(cmd, "K", 10);

        BayesianMultIdealPointOWLQN pred = new BayesianMultIdealPointOWLQN("bayesian-mult-ideal-point-owlqn");
        pred.configure(max_iters, l1, l2, K);
        pred.setAuthorVocab(debateVoteData.getAuthorVocab());
        pred.setVoteVocab(debateVoteData.getVoteVocab());

        File predFolder = new File(outputFolder, pred.getName());
        if (cmd.hasOption("train")) {
            pred.setTrain(votes, trainAuthorIndices, trainBillIndices, trainVotes);
            pred.train();
            IOUtils.createFolder(predFolder);
            pred.output(new File(predFolder, MODEL_FILE));
        }

        if (cmd.hasOption("testvote")) {
            pred.input(new File(predFolder, MODEL_FILE));
            SparseVector[] predictions = pred.test(testVotes);
            File teResultFolder = new File(predFolder, TEST_PREFIX + RESULT_FOLDER);
            IOUtils.createFolder(teResultFolder);
            AbstractModel.outputPerformances(new File(teResultFolder, RESULT_FILE),
                    AbstractVotePredictor.evaluateAll(votes, testVotes, predictions));
        }
    }

    protected void runLexicalIdealPoint(File outputFolder) {
        double rho = CLIUtils.getDoubleArgument(cmd, "rho", 0.1);
        double sigma = CLIUtils.getDoubleArgument(cmd, "sigma", 0.1);
        double lambda = CLIUtils.getDoubleArgument(cmd, "lambda", 2.5);
        double l1 = CLIUtils.getDoubleArgument(cmd, "l1", 0.0);
        double l2 = CLIUtils.getDoubleArgument(cmd, "l2", 0.0);

        LexicalIdealPoint lip = new LexicalIdealPoint();
        if (cmd.hasOption("lambda")) {
            lip.configure(outputFolder.getAbsolutePath(),
                    debateVoteData.getWordVocab().size(), rho, sigma, lambda,
                    max_iters, cmd.hasOption("tfidf"));
        } else {
            lip.configure(outputFolder.getAbsolutePath(),
                    debateVoteData.getWordVocab().size(), rho, sigma, l1, l2,
                    max_iters, cmd.hasOption("tfidf"));
        }
        File samplerFolder = new File(outputFolder, lip.getSamplerFolder());
        IOUtils.createFolder(samplerFolder);

        File trResultFolder = new File(samplerFolder, TRAIN_PREFIX + RESULT_FOLDER);
        File teResultFolder = new File(samplerFolder, TEST_PREFIX + RESULT_FOLDER);

        if (cmd.hasOption("train")) {
            IOUtils.createFolder(trResultFolder);

            lip.setupData(trainDebateIndices,
                    debateVoteData.getWords(),
                    debateVoteData.getAuthors(),
                    votes,
                    trainAuthorIndices,
                    trainBillIndices,
                    trainVotes);
            lip.initialize();
            lip.iterate();
            SparseVector[] predictions = lip.predictInMatrix();
            AbstractVotePredictor.outputPredictions(new File(trResultFolder, PREDICTION_FILE),
                    votes, predictions);
            AbstractModel.outputPerformances(new File(trResultFolder, RESULT_FILE),
                    AbstractVotePredictor.evaluateAll(votes, trainVotes, predictions));
            lip.outputState(new File(samplerFolder, MODEL_FILE));
            lip.outputLexicalRegressionParameters(new File(samplerFolder, "lexical.txt"),
                    debateVoteData.getWordVocab());
            lip.outputAuthorFeatures(new File(trResultFolder, "author.features"));

            outputAuthorScore(new File(samplerFolder, AuthorScoreFile),
                    debateVoteData.getAuthorVocab(),
                    trainAuthorIndices,
                    trainVotes,
                    lip.getUs(),
                    debateVoteData.getAuthorTable());
            outputVoteScores(new File(samplerFolder, VoteScoreFile),
                    debateVoteData.getVoteVocab(),
                    trainBillIndices,
                    lip.getXs(), lip.getYs(),
                    debateVoteData.getVoteTable());
        }

        if (cmd.hasOption("testvote")) {
            IOUtils.createFolder(teResultFolder);

            lip.setupData(testDebateIndices,
                    debateVoteData.getWords(),
                    debateVoteData.getAuthors(),
                    null,
                    testAuthorIndices,
                    testBillIndices,
                    testVotes);
            lip.inputModel(new File(samplerFolder, MODEL_FILE).toString());
            SparseVector[] predictions = lip.predictInMatrix();
            AbstractVotePredictor.outputPredictions(new File(teResultFolder, PREDICTION_FILE),
                    votes, predictions);
            AbstractModel.outputPerformances(new File(teResultFolder, RESULT_FILE),
                    AbstractVotePredictor.evaluateAll(votes, testVotes, predictions));
        }

        if (cmd.hasOption("testauthor")) {
            IOUtils.createFolder(teResultFolder);

            lip.setupData(testDebateIndices,
                    debateVoteData.getWords(),
                    debateVoteData.getAuthors(),
                    null,
                    testAuthorIndices,
                    testBillIndices,
                    testVotes);
            lip.inputModel(new File(samplerFolder, MODEL_FILE).toString());
            SparseVector[] predictions = lip.predictOutMatrix();
            AbstractVotePredictor.outputPredictions(new File(teResultFolder, PREDICTION_FILE),
                    votes, predictions);
            AbstractModel.outputPerformances(new File(teResultFolder, RESULT_FILE),
                    AbstractVotePredictor.evaluateAll(votes, testVotes, predictions));

            lip.outputAuthorFeatures(new File(teResultFolder, "author.features"));
            outputAuthorScore(new File(teResultFolder, AuthorScoreFile),
                    debateVoteData.getAuthorVocab(),
                    testAuthorIndices,
                    testVotes,
                    lip.getPredictedUs(),
                    debateVoteData.getAuthorTable());
        }

        if (cmd.hasOption("analyzeerror")) {
            analyzeError(samplerFolder);
        }

        if (cmd.hasOption("dev")) {
            if (verbose) {
                logln("Tuning " + lip.getBasename());
            }

            double[] rhos = {0.1};
            double[] sigmas = {0.1};
            double[] lambdas = {2.0, 5.0, 10.0, 25.0, 50.0, 100.0};
            double[] l1s = {1E-7, 1E-6, 1E-5, 1E-4, 1E-3, 1E-2, 1E-1, 0.0};
            double[] l2s = {1E-7, 1E-6, 1E-5, 1E-4, 1E-3, 1E-2, 1E-1, 0.0};

            File devFolder = new File(outputFolder, lip.getBasename());
            IOUtils.createFolder(devFolder);
            boolean hasHeader = false;
            try {
//                BufferedWriter lWriter = IOUtils.getBufferedWriter(new File(devFolder, "dev-lbfgs.txt"));
//                for (double r : rhos) {
//                    for (double s : sigmas) {
//                        for (double l : lambdas) {
//                            if (verbose) {
//                                logln("--- Dev L-LBFGS: rho = " + r
//                                        + ". sigma = " + s
//                                        + ". lambda = " + l);
//                            }
//
//                            lip = new LexicalIdealPoint();
//                            lip.configure(outputFolder.getAbsolutePath(),
//                                    debateVoteData.getWordVocab().size(), r, s, l,
//                                    max_iters, cmd.hasOption("tfidf"));
//                            samplerFolder = new File(outputFolder, lip.getSamplerFolder());
//                            teResultFolder = new File(samplerFolder, TEST_PREFIX + RESULT_FOLDER);
//                            IOUtils.createFolder(teResultFolder);
//
//                            // train
//                            if (verbose) {
//                                logln("--- --- Training ...");
//                            }
//                            lip.setupData(trainDebateIndices,
//                                    debateVoteData.getWords(),
//                                    debateVoteData.getAuthors(),
//                                    votes,
//                                    trainAuthorIndices,
//                                    trainBillIndices,
//                                    trainVotes);
//                            lip.initialize();
//                            lip.iterate();
//                            lip.outputState(new File(samplerFolder, MODEL_FILE));
//
//                            // test
//                            if (verbose) {
//                                logln("--- --- Testing ...");
//                            }
//                            lip.setupData(testDebateIndices,
//                                    debateVoteData.getWords(),
//                                    debateVoteData.getAuthors(),
//                                    null,
//                                    testAuthorIndices,
//                                    testBillIndices,
//                                    testVotes);
//                            lip.inputModel(new File(samplerFolder, MODEL_FILE).toString());
//                            SparseVector[] predictions = lip.predictOutMatrix();
//                            ArrayList<Measurement> results = AbstractVotePredictor.
//                                    evaluateAll(votes, testVotes, predictions);
//                            AbstractModel.outputPerformances(new File(teResultFolder, RESULT_FILE), results);
//
//                            if (!hasHeader) {
//                                lWriter.write("rho\tsigma\tlambda");
//                                for (Measurement m : results) {
//                                    lWriter.write("\t" + m.getName());
//                                }
//                                lWriter.write("\n");
//                                hasHeader = true;
//                            }
//
//                            lWriter.write(r + "\t" + s + "\t" + l);
//                            for (Measurement m : results) {
//                                lWriter.write("\t" + m.getValue());
//                            }
//                            lWriter.write("\n");
//                        }
//                    }
//                }
//                lWriter.close();

                BufferedWriter oWriter = IOUtils.getBufferedWriter(new File(devFolder, "dev-owlqn.txt"));
                for (double r : rhos) {
                    for (double s : sigmas) {
                        for (double el1 : l1s) {
                            for (double el2 : l2s) {
                                if (verbose) {
                                    logln("--- Dev OWL-QN: rho = " + r
                                            + ". sigma = " + s
                                            + ". l1 = " + l1
                                            + ". l2 = " + l2);
                                }

                                lip = new LexicalIdealPoint();
                                lip.configure(outputFolder.getAbsolutePath(),
                                        debateVoteData.getWordVocab().size(), r, s, el1, el2,
                                        max_iters, cmd.hasOption("tfidf"));
                                samplerFolder = new File(outputFolder, lip.getSamplerFolder());
                                teResultFolder = new File(samplerFolder, TEST_PREFIX + RESULT_FOLDER);
                                IOUtils.createFolder(teResultFolder);

                                // train
                                if (verbose) {
                                    logln("--- --- Training ...");
                                }
                                lip.setupData(trainDebateIndices,
                                        debateVoteData.getWords(),
                                        debateVoteData.getAuthors(),
                                        votes,
                                        trainAuthorIndices,
                                        trainBillIndices,
                                        trainVotes);
                                lip.initialize();
                                lip.iterate();
                                lip.outputState(new File(samplerFolder, MODEL_FILE));

                                // test
                                if (verbose) {
                                    logln("--- --- Testing ...");
                                }
                                lip.setupData(testDebateIndices,
                                        debateVoteData.getWords(),
                                        debateVoteData.getAuthors(),
                                        null,
                                        testAuthorIndices,
                                        testBillIndices,
                                        testVotes);
                                lip.inputModel(new File(samplerFolder, MODEL_FILE).toString());
                                SparseVector[] predictions = lip.predictOutMatrix();
                                ArrayList<Measurement> results = AbstractVotePredictor.
                                        evaluateAll(votes, testVotes, predictions);
                                AbstractModel.outputPerformances(new File(teResultFolder, RESULT_FILE), results);

                                if (!hasHeader) {
                                    oWriter.write("rho\tsigma\tl1\tl2");
                                    for (Measurement m : results) {
                                        oWriter.write("\t" + m.getName());
                                    }
                                    oWriter.write("\n");
                                    hasHeader = true;
                                }

                                oWriter.write(r + "\t" + s + "\t" + el1 + "\t" + el2);
                                for (Measurement m : results) {
                                    oWriter.write("\t" + m.getValue());
                                }
                                oWriter.write("\n");
                            }
                        }
                    }
                }
                oWriter.close();
            } catch (Exception e) {
                e.printStackTrace();
                throw new RuntimeException();
            }
        }
    }

    protected void runSLDAIdealPoint(File outputFolder) {
        int K;
        double[][] issuePhis;
        if (cmd.hasOption("K")) {
            issuePhis = null;
            K = Integer.parseInt(cmd.getOptionValue("K"));
        } else {
            issuePhis = estimateIssues();
            K = issuePhis.length;
        }
        double alpha = CLIUtils.getDoubleArgument(cmd, "alpha", 0.1);
        double beta = CLIUtils.getDoubleArgument(cmd, "beta", 0.1);
        double mu = CLIUtils.getDoubleArgument(cmd, "mu", 0.0);
        double sigma = CLIUtils.getDoubleArgument(cmd, "sigma", 10);
        double rate_alpha = CLIUtils.getDoubleArgument(cmd, "rate-alpha", 1);
        double rate_eta = CLIUtils.getDoubleArgument(cmd, "rate-eta", 0.01);
        double rho = CLIUtils.getDoubleArgument(cmd, "rho", 1.0);

        SLDAIdealPoint sampler = new SLDAIdealPoint();
        sampler.setVerbose(verbose);
        sampler.setDebug(debug);
        sampler.setLog(true);
        sampler.setReport(true);
        sampler.setWordVocab(debateVoteData.getWordVocab());
        sampler.setAuthorVocab(debateVoteData.getAuthorVocab());
        sampler.setVoteVocab(debateVoteData.getVoteVocab());

        sampler.configure(outputFolder.getAbsolutePath(),
                debateVoteData.getWordVocab().size(), K,
                alpha, beta, rho, mu, sigma, rate_alpha, rate_eta,
                initState, paramOpt,
                burn_in, max_iters, sample_lag, report_interval);
        File samplerFolder = new File(outputFolder, sampler.getSamplerFolder());
        IOUtils.createFolder(samplerFolder);

        if (cmd.hasOption("train")) {
            sampler.setupData(trainDebateIndices,
                    debateVoteData.getWords(),
                    debateVoteData.getAuthors(),
                    votes,
                    trainAuthorIndices,
                    trainBillIndices,
                    trainVotes);
            sampler.initialize(issuePhis);
            sampler.iterate();
            if (issuePhis == null) {
                sampler.outputTopicTopWords(new File(samplerFolder, TopWordFile),
                        numTopWords);
            } else {
                sampler.outputTopicTopWords(new File(samplerFolder, TopWordFile),
                        numTopWords, policyAgendaIssues);
            }

            outputAuthorScore(new File(samplerFolder, AuthorScoreFile),
                    debateVoteData.getAuthorVocab(),
                    trainAuthorIndices,
                    trainVotes,
                    sampler.getUs(),
                    debateVoteData.getAuthorTable());
            outputVoteScores(new File(samplerFolder, VoteScoreFile),
                    debateVoteData.getVoteVocab(),
                    trainBillIndices,
                    sampler.getXs(), sampler.getYs(),
                    debateVoteData.getVoteTable());

            File trResultFolder = new File(samplerFolder, TRAIN_PREFIX + RESULT_FOLDER);
            IOUtils.createFolder(trResultFolder);
            SparseVector[] predictions = sampler.predictInMatrix();
            AbstractVotePredictor.outputPredictions(new File(trResultFolder, PREDICTION_FILE),
                    votes, predictions);
            AbstractModel.outputPerformances(new File(trResultFolder, RESULT_FILE),
                    AbstractVotePredictor.evaluateAll(votes, trainVotes, predictions));
        }

        if (cmd.hasOption("testvote")) { // average over multiple test chains
            sampler.setupData(testDebateIndices,
                    debateVoteData.getWords(),
                    debateVoteData.getAuthors(),
                    null,
                    testAuthorIndices,
                    testBillIndices,
                    testVotes);
            sampler.inputFinalState();
            SparseVector[] predictions = sampler.predictInMatrixMultiples();

            File teResultFolder = new File(samplerFolder, TEST_PREFIX + RESULT_FOLDER);
            IOUtils.createFolder(teResultFolder);
            AbstractVotePredictor.outputPredictions(new File(teResultFolder, PREDICTION_FILE),
                    votes, predictions);
            AbstractModel.outputPerformances(new File(teResultFolder, RESULT_FILE),
                    AbstractVotePredictor.evaluateAll(votes, testVotes, predictions));
        }

        if (cmd.hasOption("testauthor")) {
            File teResultFolder = new File(samplerFolder, TEST_PREFIX + RESULT_FOLDER);
            IOUtils.createFolder(teResultFolder);

            SparseVector[] predictions;
            if (cmd.hasOption("parallel")) {
                predictions = SLDAIdealPoint.parallelTest(testDebateIndices,
                        debateVoteData.getWords(),
                        debateVoteData.getAuthors(),
                        testAuthorIndices,
                        testVotes,
                        new File(samplerFolder, AbstractSampler.IterPredictionFolder),
                        sampler);
            } else {
                sampler.setupData(testDebateIndices,
                        debateVoteData.getWords(),
                        debateVoteData.getAuthors(),
                        null,
                        testAuthorIndices,
                        testBillIndices,
                        testVotes);
                predictions = sampler.test(null, null,
                        new File(samplerFolder, TEST_PREFIX + AssignmentFile));
                outputAuthorScore(new File(teResultFolder, AuthorScoreFile),
                        debateVoteData.getAuthorVocab(),
                        testAuthorIndices,
                        testVotes,
                        sampler.getPredictedUs(),
                        debateVoteData.getAuthorTable());
            }
            AbstractVotePredictor.outputPredictions(new File(teResultFolder, PREDICTION_FILE),
                    votes, predictions);
            AbstractModel.outputPerformances(new File(teResultFolder, RESULT_FILE),
                    AbstractVotePredictor.evaluateAll(votes, testVotes, predictions));
        }

        if (cmd.hasOption("analyzeerror")) {
            analyzeError(samplerFolder);
        }
    }

    protected void runHierSingleTIPM(File outputFolder) {
        int K;
        double[][] issuePhis;
        if (cmd.hasOption("K")) {
            issuePhis = null;
            K = Integer.parseInt(cmd.getOptionValue("K"));
        } else {
            issuePhis = estimateIssues();
            K = issuePhis.length;
        }
        int J = CLIUtils.getIntegerArgument(cmd, "J", 5);
        double topicAlpha = CLIUtils.getDoubleArgument(cmd, "alpha", 0.1);
        double topicBeta = CLIUtils.getDoubleArgument(cmd, "beta", 0.1);
        double frameAlpha = topicAlpha;
        double frameBeta = topicAlpha;

        double rho = CLIUtils.getDoubleArgument(cmd, "rho", 0.1);
        double sigma = CLIUtils.getDoubleArgument(cmd, "sigma", 0.5);
        double gamma = CLIUtils.getDoubleArgument(cmd, "gamma", 2.5);

        HierSingleTIPM sampler = new HierSingleTIPM();
        sampler.setVerbose(verbose);
        sampler.setDebug(debug);
        sampler.setLog(true);
        sampler.setReport(true);
        sampler.setWordVocab(debateVoteData.getWordVocab());
        sampler.setAuthorVocab(debateVoteData.getAuthorVocab());
        sampler.setVoteVocab(debateVoteData.getVoteVocab());

        sampler.configure(outputFolder.getAbsolutePath(),
                debateVoteData.getWordVocab().size(), K, J,
                topicAlpha, frameAlpha, topicBeta, frameBeta,
                rho, sigma, gamma, issuePhis,
                initState, paramOpt,
                burn_in, max_iters, sample_lag, report_interval);
        File samplerFolder = new File(outputFolder, sampler.getSamplerFolder());
        IOUtils.createFolder(samplerFolder);

        if (cmd.hasOption("train")) {
            sampler.setupData(trainDebateIndices,
                    debateVoteData.getWords(),
                    debateVoteData.getAuthors(),
                    votes,
                    trainAuthorIndices,
                    trainBillIndices,
                    trainVotes);
            sampler.setTopicVocab(policyAgendaIssues);
            sampler.initialize();

            outputAuthorScore(new File(samplerFolder, AuthorScoreFile + ".init"),
                    debateVoteData.getAuthorVocab(),
                    trainAuthorIndices,
                    trainVotes,
                    sampler.getUs(),
                    debateVoteData.getAuthorTable());

            outputVoteScores(new File(samplerFolder, VoteScoreFile + ".init"),
                    debateVoteData.getVoteVocab(),
                    trainBillIndices,
                    sampler.getXs(), sampler.getYs(),
                    debateVoteData.getVoteTable());

            sampler.metaIterate();

            if (issuePhis == null) {
                sampler.outputTopicTopWords(new File(samplerFolder, TopWordFile),
                        numTopWords);
            } else {
                sampler.outputTopicTopWords(new File(samplerFolder, TopWordFile),
                        numTopWords, policyAgendaIssues);
            }

            outputAuthorScore(new File(samplerFolder, AuthorScoreFile),
                    debateVoteData.getAuthorVocab(),
                    trainAuthorIndices,
                    trainVotes,
                    sampler.getUs(),
                    debateVoteData.getAuthorTable());

            outputAuthorScore(new File(samplerFolder, AuthorScoreFile + ".predicted"),
                    debateVoteData.getAuthorVocab(),
                    trainAuthorIndices,
                    trainVotes,
                    sampler.getPredictedUs(),
                    debateVoteData.getAuthorTable());

            outputAuthorScore(new File(samplerFolder, AuthorScoreFile + ".multiple"),
                    debateVoteData.getAuthorVocab(),
                    trainAuthorIndices,
                    trainVotes,
                    sampler.getUs(),
                    sampler.getMultipleUs(),
                    debateVoteData.getAuthorTable());

            outputVoteScores(new File(samplerFolder, VoteScoreFile),
                    debateVoteData.getVoteVocab(),
                    trainBillIndices,
                    sampler.getXs(), sampler.getYs(),
                    debateVoteData.getVoteTable());

            File trResultFolder = new File(samplerFolder, TRAIN_PREFIX + RESULT_FOLDER);
            IOUtils.createFolder(trResultFolder);
            SparseVector[] predictions = sampler.predictInMatrix();
            AbstractVotePredictor.outputPredictions(new File(trResultFolder, PREDICTION_FILE),
                    votes, predictions);
            AbstractModel.outputPerformances(new File(trResultFolder, RESULT_FILE),
                    AbstractVotePredictor.evaluateAll(votes, trainVotes, predictions));
        }

        if (cmd.hasOption("analyze")) {
            sampler.setAuthorVocab(debateVoteData.getAuthorVocab());
            sampler.setupData(trainDebateIndices,
                    debateVoteData.getWords(),
                    debateVoteData.getAuthors(),
                    votes,
                    trainAuthorIndices,
                    trainBillIndices,
                    trainVotes);
            sampler.setTopicVocab(policyAgendaIssues);

            File analysisFolder = new File(samplerFolder, "analysis");
            IOUtils.createFolder(analysisFolder);
            sampler.loadEtaList();
            sampler.inputFinalState();

            sampler.outputTheta(new File(analysisFolder, "theta.txt"));
            sampler.outputTopicFrameVariance(new File(analysisFolder, "eta_variances.txt"));
            sampler.outputTopicTopWords(new File(analysisFolder, "top-words.txt"), numTopWords);
            sampler.outputHierarchyWithDetails(new File(analysisFolder, "top-words-details.txt"),
                    debateVoteData.getAuthorTable(),
                    debateVoteData.getDocIds(),
                    debateVoteData.getRawSentences());

        }
    }

    protected double[][] createBillPriors() {
        int K = policyAgendaIssues.size();
        double[][] billTopicPriors = new double[trainVoteTopics.length][K];
        double boost = 10;
        for (int bb = 0; bb < trainVoteTopics.length; bb++) {
            int oriTopicCode = trainVoteTopics[bb];
            String oriTopicStr = billData.getTopicVocab().get(oriTopicCode);
            int idx = policyAgendaIssues.indexOf(oriTopicStr);

            System.out.println(oriTopicCode
                    + ". " + oriTopicStr
                    + ". " + idx
                    + ". " + (idx < 0 ? "NA" : policyAgendaIssues.get(idx)));

            if (idx >= 0) {
                for (int kk = 0; kk < K; kk++) {
                    if (kk == idx) {
                        billTopicPriors[bb][kk] = (1.0 + boost) / (K + boost);
                    } else {
                        billTopicPriors[bb][kk] = 1.0 / (K + boost);
                    }
                }
            } else {
                Arrays.fill(billTopicPriors[bb], 1.0 / K);
            }
        }

        return billTopicPriors;
    }

    protected void runHierMultTIPM(File outputFolder) {
        createBillPriors();
        int K;
        double[][] issuePhis;
        if (cmd.hasOption("K")) {
            issuePhis = null;
            K = Integer.parseInt(cmd.getOptionValue("K"));
        } else {
            issuePhis = estimateIssues();
            K = issuePhis.length;
        }
        int J = CLIUtils.getIntegerArgument(cmd, "J", 5);
        double topicAlpha = CLIUtils.getDoubleArgument(cmd, "alpha", 0.1);
        double topicBeta = CLIUtils.getDoubleArgument(cmd, "beta", 0.1);
        double frameAlpha = topicAlpha;
        double frameBeta = topicAlpha;

        double rho = CLIUtils.getDoubleArgument(cmd, "rho", 0.1);
        double sigma = CLIUtils.getDoubleArgument(cmd, "sigma", 0.5);
        double gamma = CLIUtils.getDoubleArgument(cmd, "gamma", 2.5);
        double lambda = CLIUtils.getDoubleArgument(cmd, "lambda", 0.75);
        double epsilon = CLIUtils.getDoubleArgument(cmd, "epsilon", 0.001);

        HierMultiTIPM sampler = new HierMultiTIPM();
        sampler.setVerbose(verbose);
        sampler.setDebug(debug);
        sampler.setLog(true);
        sampler.setReport(true);
        sampler.setWordVocab(debateVoteData.getWordVocab());
        sampler.setAuthorVocab(debateVoteData.getAuthorVocab());
        sampler.setVoteVocab(debateVoteData.getVoteVocab());

        sampler.configure(outputFolder.getAbsolutePath(),
                debateVoteData.getWordVocab().size(), K, J,
                topicAlpha, frameAlpha, topicBeta, frameBeta,
                rho, sigma, gamma, lambda, epsilon, issuePhis,
                initState, paramOpt,
                burn_in, max_iters, sample_lag, report_interval);
        File samplerFolder = new File(outputFolder, sampler.getSamplerFolder());
        IOUtils.createFolder(samplerFolder);

        if (cmd.hasOption("train")) {
            sampler.setupData(trainDebateIndices,
                    debateVoteData.getWords(),
                    debateVoteData.getAuthors(),
                    votes,
                    trainAuthorIndices,
                    trainBillIndices,
                    trainVotes);
            sampler.setBillWords(trainVoteWords);
            sampler.setTopicVocab(policyAgendaIssues);
            double[][] billTopicPriors = createBillPriors();
            sampler.setBillTopicPriors(billTopicPriors);

            outputVoteScores(new File(samplerFolder, VoteScoreFile + ".topics.priors.init"),
                    debateVoteData.getVoteVocab(),
                    trainBillIndices,
                    billTopicPriors,
                    null, null,
                    debateVoteData.getVoteTable());

            sampler.initialize();

            outputVoteScores(new File(samplerFolder, VoteScoreFile + ".init"),
                    debateVoteData.getVoteVocab(),
                    trainBillIndices,
                    sampler.getXs(), sampler.getYs(),
                    debateVoteData.getVoteTable());

            outputVoteScores(new File(samplerFolder, VoteScoreFile + ".topics.init"),
                    debateVoteData.getVoteVocab(),
                    trainBillIndices,
                    sampler.getBillThetas(),
                    sampler.getXs(), sampler.getYs(),
                    debateVoteData.getVoteTable());

            outputAuthorScore(new File(samplerFolder, AuthorScoreFile + ".init"),
                    debateVoteData.getAuthorVocab(),
                    trainAuthorIndices,
                    trainVotes,
                    sampler.getUs(),
                    sampler.getMultiUs(),
                    debateVoteData.getAuthorTable());

            sampler.metaIterate();

            if (issuePhis == null) {
                sampler.outputTopicTopWords(new File(samplerFolder, TopWordFile),
                        numTopWords);
            } else {
                sampler.outputTopicTopWords(new File(samplerFolder, TopWordFile),
                        numTopWords, policyAgendaIssues);
            }

            outputAuthorScore(new File(samplerFolder, AuthorScoreFile),
                    debateVoteData.getAuthorVocab(),
                    trainAuthorIndices,
                    trainVotes,
                    sampler.getUs(),
                    sampler.getMultiUs(),
                    debateVoteData.getAuthorTable());

            outputAuthorScore(new File(samplerFolder, AuthorScoreFile + ".estimate"),
                    debateVoteData.getAuthorVocab(),
                    trainAuthorIndices,
                    trainVotes,
                    sampler.getUs(),
                    sampler.getEstimatedUs(),
                    debateVoteData.getAuthorTable());

            outputVoteScores(new File(samplerFolder, VoteScoreFile),
                    debateVoteData.getVoteVocab(),
                    trainBillIndices,
                    sampler.getXs(), sampler.getYs(),
                    debateVoteData.getVoteTable());

            File trResultFolder = new File(samplerFolder, TRAIN_PREFIX + RESULT_FOLDER);
            IOUtils.createFolder(trResultFolder);
            SparseVector[] predictions = sampler.predictInMatrix();
            AbstractVotePredictor.outputPredictions(new File(trResultFolder, PREDICTION_FILE),
                    votes, predictions);
            AbstractModel.outputPerformances(new File(trResultFolder, RESULT_FILE),
                    AbstractVotePredictor.evaluateAll(votes, trainVotes, predictions));
        }

        if (cmd.hasOption("analyze")) {
            sampler.setAuthorVocab(debateVoteData.getAuthorVocab());
            sampler.setupData(trainDebateIndices,
                    debateVoteData.getWords(),
                    debateVoteData.getAuthors(),
                    votes,
                    trainAuthorIndices,
                    trainBillIndices,
                    trainVotes);
            sampler.setTopicVocab(policyAgendaIssues);
            sampler.loadEtaList();

            File analysisFolder = new File(samplerFolder, "analysis");
            IOUtils.createFolder(analysisFolder);
            sampler.inputFinalState();

            sampler.outputTheta(new File(analysisFolder, "theta.txt"));
            sampler.outputTopicFrameVariance(new File(analysisFolder, "eta_variances.txt"));
            sampler.outputTopicTopWords(new File(analysisFolder, "top-words.txt"), numTopWords);
            sampler.outputHierarchyWithDetails(new File(analysisFolder, "top-words-details.txt"),
                    debateVoteData.getAuthorTable(),
                    debateVoteData.getDocIds(),
                    debateVoteData.getRawSentences());
        }

        if (cmd.hasOption("testauthor")) {
            File teResultFolder = new File(samplerFolder, TEST_PREFIX + RESULT_FOLDER);
            IOUtils.createFolder(teResultFolder);

            SparseVector[] predictions;
            if (cmd.hasOption("parallel")) {
                predictions = HierMultiTIPM.parallelTest(testDebateIndices,
                        debateVoteData.getWords(),
                        debateVoteData.getAuthors(),
                        testAuthorIndices,
                        testVotes,
                        new File(samplerFolder, AbstractSampler.IterPredictionFolder),
                        sampler);
            } else {
                sampler.setupData(testDebateIndices,
                        debateVoteData.getWords(),
                        debateVoteData.getAuthors(),
                        null,
                        testAuthorIndices,
                        testBillIndices,
                        testVotes);
                predictions = sampler.test(null, null,
                        new File(samplerFolder, TEST_PREFIX + AssignmentFile));
            }
            AbstractVotePredictor.outputPredictions(new File(teResultFolder, PREDICTION_FILE),
                    votes, predictions);
            AbstractModel.outputPerformances(new File(teResultFolder, RESULT_FILE),
                    AbstractVotePredictor.evaluateAll(votes, testVotes, predictions));
        }
    }

    protected void runMultTopicIdealPoint(File outputFolder) {
        double[][] billTopicPriors = createBillPriors();
        int K = billTopicPriors.length;

        double alpha = CLIUtils.getDoubleArgument(cmd, "alpha", 5.0);
        double eta = CLIUtils.getDoubleArgument(cmd, "eta", 0.01);
        double mu = CLIUtils.getDoubleArgument(cmd, "mu", 0.0);
        double sigma = CLIUtils.getDoubleArgument(cmd, "sigma", 2.5);

        MultTopicIdealPoint pred = new MultTopicIdealPoint("topic-ideal-point");
        pred.configure(alpha, eta, max_iters, mu, sigma);
        pred.setAuthorVocab(debateVoteData.getAuthorVocab());
        pred.setVoteVocab(debateVoteData.getVoteVocab());

        File predFolder = new File(outputFolder, pred.getName());

        if (cmd.hasOption("train")) {
            pred.setTrain(votes, trainAuthorIndices, trainBillIndices, trainVotes);
            pred.train();
            IOUtils.createFolder(predFolder);
            pred.output(new File(predFolder, MODEL_FILE));
            outputAuthorScore(new File(predFolder, AuthorScoreFile),
                    debateVoteData.getAuthorVocab(),
                    trainAuthorIndices,
                    trainVotes,
                    pred.getUs(),
                    debateVoteData.getAuthorTable());
            outputVoteScores(new File(predFolder, VoteScoreFile),
                    debateVoteData.getVoteVocab(),
                    trainBillIndices,
                    pred.getXs(), pred.getYs(),
                    debateVoteData.getVoteTable());

            // evaluate on training data
//            SparseVector[] predictions = pred.test(trainVotes);
//            File trResultFolder = new File(predFolder, TRAIN_PREFIX + RESULT_FOLDER);
//            IOUtils.createFolder(trResultFolder);
//            AbstractModel.outputPerformances(new File(trResultFolder, RESULT_FILE),
//                    AbstractVotePredictor.evaluateAll(votes, trainVotes, predictions));
        }

    }

    protected void runHierMultSHDP(File outputFolder) {
        createBillPriors();
        int K;
        double[][] issuePhis;
        if (cmd.hasOption("K")) {
            issuePhis = null;
            K = Integer.parseInt(cmd.getOptionValue("K"));
        } else {
            issuePhis = estimateIssues();
            K = issuePhis.length;
        }
        int J = CLIUtils.getIntegerArgument(cmd, "J", 0);
        double topicAlpha = CLIUtils.getDoubleArgument(cmd, "alpha", 0.1);
        double frameAlphaGlobal = CLIUtils.getDoubleArgument(cmd, "global-alpha", 0.001);
        double frameAlphaLocal = CLIUtils.getDoubleArgument(cmd, "local-alpha", 0.1);

        double topicBeta = CLIUtils.getDoubleArgument(cmd, "topic-beta", 0.1);
        double frameBeta = CLIUtils.getDoubleArgument(cmd, "frame-beta", 0.1);

        double rho = CLIUtils.getDoubleArgument(cmd, "rho", 0.1);
        double sigma = CLIUtils.getDoubleArgument(cmd, "sigma", 0.5);
        double gamma = CLIUtils.getDoubleArgument(cmd, "gamma", 2.5);
        double lambda = CLIUtils.getDoubleArgument(cmd, "lambda", 0.5);
        double epsilon = CLIUtils.getDoubleArgument(cmd, "epsilon", 0.001);
        int initMaxIter = CLIUtils.getIntegerArgument(cmd, "init-maxiter", 5000);
        boolean isUP = cmd.hasOption("up");

        HierMultSHDP sampler = new HierMultSHDP();
        sampler.setVerbose(verbose);
        sampler.setDebug(debug);
        sampler.setLog(true);
        sampler.setReport(true);
        sampler.setWordVocab(debateVoteData.getWordVocab());
        sampler.setAuthorVocab(debateVoteData.getAuthorVocab());
        sampler.setVoteVocab(debateVoteData.getVoteVocab());
        sampler.setTopicVocab(policyAgendaIssues);
        sampler.setBillWords(trainVoteWords);
        sampler.setInitMaxIter(initMaxIter);

        PathAssumption pathAssumption = PathAssumption.MAXIMAL;
        String path = CLIUtils.getStringArgument(cmd, "path", "max");
        switch (path) {
            case "max":
                pathAssumption = PathAssumption.MAXIMAL;
                break;
            case "min":
                pathAssumption = PathAssumption.MINIMAL;
                break;
            case "uniproc":
                pathAssumption = PathAssumption.UNIPROC;
                break;
            case "antoniak":
                pathAssumption = PathAssumption.ANTONIAK;
                break;
            default:
                throw new RuntimeException("Path assumption " + path + " not supported");
        }
        int threshold = 25;
        sampler.configure(outputFolder.getAbsolutePath(),
                debateVoteData.getWordVocab().size(), K, J,
                topicAlpha, frameAlphaGlobal, frameAlphaLocal, topicBeta, frameBeta,
                rho, sigma, gamma, lambda, epsilon, threshold, pathAssumption, issuePhis,
                initState, paramOpt,
                burn_in, max_iters, sample_lag, report_interval);
        File samplerFolder = new File(outputFolder, sampler.getSamplerFolder());
        IOUtils.createFolder(samplerFolder);

        double[][] billTopicPriors = createBillPriors();

        HashMap<Integer, Boolean> teapartycaucus = new HashMap<>();
        for (int aa : trainAuthorIndices) {
            String aid = debateVoteData.getAuthorVocab().get(aa);
            Author author = debateVoteData.getAuthorTable().get(aid);
            String icpsrid = author.getProperty(GTLegislator.ICPSRID);
            boolean tp = false;
            Integer tpc = teapartyCaucusMapping.get(icpsrid);
            if (tpc != null && tpc == 1) {
                tp = true;
            }
            teapartycaucus.put(aa, tp);
        }
        int count = 0;
        for (int aa : teapartycaucus.keySet()) {
            if (teapartycaucus.get(aa)) {
                count++;
            }
        }
        System.out.println("# Tea Party Caucus members = " + count);

        if (cmd.hasOption("initialize")) {
            sampler.setupData(trainDebateIndices,
                    debateVoteData.getWords(),
                    debateVoteData.getAuthors(),
                    votes,
                    trainAuthorIndices,
                    trainBillIndices,
                    trainVotes);

            sampler.setBillTopicPriors(billTopicPriors);

            outputVoteScores(new File(samplerFolder, VoteScoreFile + ".topics.priors.init"),
                    debateVoteData.getVoteVocab(),
                    trainBillIndices,
                    billTopicPriors,
                    null, null,
                    debateVoteData.getVoteTable());

            sampler.initialize();

            outputVoteScores(new File(samplerFolder, VoteScoreFile + ".init"),
                    debateVoteData.getVoteVocab(),
                    trainBillIndices,
                    sampler.getXs(), sampler.getYs(),
                    debateVoteData.getVoteTable());

            outputVoteScores(new File(samplerFolder, VoteScoreFile + ".topics.init"),
                    debateVoteData.getVoteVocab(),
                    trainBillIndices,
                    sampler.getBillThetas(),
                    sampler.getXs(), sampler.getYs(),
                    debateVoteData.getVoteTable());

            outputAuthorScore(new File(samplerFolder, AuthorScoreFile + ".init"),
                    debateVoteData.getAuthorVocab(),
                    trainAuthorIndices,
                    trainVotes,
                    sampler.getUs(),
                    sampler.getMultiUs(),
                    debateVoteData.getAuthorTable());

            outputVoteScores(new File(samplerFolder, VoteScoreFile + ".top-topics.init"),
                    debateVoteData.getVoteVocab(),
                    trainBillIndices,
                    sampler.getBillThetas(),
                    debateVoteData.getVoteTable(), 5);
        }

        if (cmd.hasOption("train")) {
            sampler.setupData(trainDebateIndices,
                    debateVoteData.getWords(),
                    debateVoteData.getAuthors(),
                    votes,
                    trainAuthorIndices,
                    trainBillIndices,
                    trainVotes);
            sampler.setBillTopicPriors(billTopicPriors);

            outputVoteScores(new File(samplerFolder, VoteScoreFile + ".topics.priors.init"),
                    debateVoteData.getVoteVocab(),
                    trainBillIndices,
                    billTopicPriors,
                    null, null,
                    debateVoteData.getVoteTable());

            sampler.initialize();

            outputVoteScores(new File(samplerFolder, VoteScoreFile + ".init"),
                    debateVoteData.getVoteVocab(),
                    trainBillIndices,
                    sampler.getXs(), sampler.getYs(),
                    debateVoteData.getVoteTable());

            outputVoteScores(new File(samplerFolder, VoteScoreFile + ".topics.init"),
                    debateVoteData.getVoteVocab(),
                    trainBillIndices,
                    sampler.getBillThetas(),
                    sampler.getXs(), sampler.getYs(),
                    debateVoteData.getVoteTable());

            outputAuthorScore(new File(samplerFolder, AuthorScoreFile + ".init"),
                    debateVoteData.getAuthorVocab(),
                    trainAuthorIndices,
                    trainVotes,
                    sampler.getUs(),
                    sampler.getMultiUs(),
                    debateVoteData.getAuthorTable());

            outputVoteScores(new File(samplerFolder, VoteScoreFile + ".top-topics.init"),
                    debateVoteData.getVoteVocab(),
                    trainBillIndices,
                    sampler.getBillThetas(),
                    debateVoteData.getVoteTable(), 5);

            sampler.metaIterate();

            if (issuePhis == null) {
                sampler.outputTopicTopWords(new File(samplerFolder, TopWordFile),
                        numTopWords);
            } else {
                sampler.outputTopicTopWords(new File(samplerFolder, TopWordFile),
                        numTopWords, policyAgendaIssues);
            }

            outputAuthorScore(new File(samplerFolder, AuthorScoreFile),
                    debateVoteData.getAuthorVocab(),
                    trainAuthorIndices,
                    trainVotes,
                    sampler.getUs(),
                    sampler.getMultiUs(),
                    debateVoteData.getAuthorTable());

//            outputAuthorScore(new File(samplerFolder, AuthorScoreFile + ".estimate"),
//                    debateVoteData.getAuthorVocab(),
//                    trainAuthorIndices,
//                    trainVotes,
//                    sampler.getUs(),
//                    sampler.getEstimatedUs(),
//                    debateVoteData.getAuthorTable());
            outputAuthorScore(new File(samplerFolder, AuthorScoreFile + ".count"),
                    debateVoteData.getAuthorVocab(),
                    trainAuthorIndices,
                    trainVotes,
                    sampler.getUs(),
                    sampler.getMultiUs(),
                    sampler.getAuthorTopicCounts(),
                    debateVoteData.getAuthorTable());

            outputVoteScores(new File(samplerFolder, VoteScoreFile),
                    debateVoteData.getVoteVocab(),
                    trainBillIndices,
                    sampler.getXs(), sampler.getYs(),
                    debateVoteData.getVoteTable());

            File trResultFolder = new File(samplerFolder, TRAIN_PREFIX + RESULT_FOLDER);
            IOUtils.createFolder(trResultFolder);
            SparseVector[] predictions = sampler.predictInMatrix();
            AbstractVotePredictor.outputPredictions(new File(trResultFolder, PREDICTION_FILE),
                    votes, predictions);
            AbstractModel.outputPerformances(new File(trResultFolder, RESULT_FILE),
                    AbstractVotePredictor.evaluateAll(votes, trainVotes, predictions));
        }

        if (cmd.hasOption("analyze")) {
            sampler.setAuthorVocab(debateVoteData.getAuthorVocab());
            sampler.setupData(trainDebateIndices,
                    debateVoteData.getWords(),
                    debateVoteData.getAuthors(),
                    votes,
                    trainAuthorIndices,
                    trainBillIndices,
                    trainVotes);
            sampler.setTopicVocab(policyAgendaIssues);
            File analysisFolder = new File(samplerFolder, "analysis");
            IOUtils.createFolder(analysisFolder);
            sampler.inputFinalState();

            sampler.outputTheta(new File(analysisFolder, "theta.txt"));

            sampler.outputTopicAttentions(new File(analysisFolder, "topic-attentions.txt"),
                    teapartycaucus);

            outputAuthorScore(new File(samplerFolder, AuthorScoreFile + ".counts"),
                    debateVoteData.getAuthorVocab(),
                    trainAuthorIndices,
                    trainVotes,
                    sampler.getUs(),
                    sampler.getMultiUs(),
                    sampler.getAuthorTopicCounts(),
                    debateVoteData.getAuthorTable());

//            outputVoteScores(new File(samplerFolder, VoteScoreFile + ".top-topics.init"),
//                    debateVoteData.getVoteVocab(),
//                    trainBillIndices,
//                    sampler.getBillThetas(),
//                    debateVoteData.getVoteTable(), 5);
            outputAuthorScore(new File(samplerFolder, AuthorScoreFile),
                    debateVoteData.getAuthorVocab(),
                    trainAuthorIndices,
                    trainVotes,
                    sampler.getUs(),
                    sampler.getMultiUs(),
                    debateVoteData.getAuthorTable());

//            sampler.outputTopicFrameVariance(new File(analysisFolder, "eta_variances.txt"));
//            sampler.outputTopicTopWords(new File(analysisFolder, "top-words.txt"), numTopWords);
            File detailTopWordFile = new File(analysisFolder, "top-words-details.txt");
            File subtreeFolder = new File(analysisFolder, "subtrees");
            IOUtils.createFolder(subtreeFolder);
            sampler.outputHierarchyWithDetails(detailTopWordFile, subtreeFolder,
                    debateVoteData.getAuthorTable(),
                    debateVoteData.getDocIds(),
                    debateVoteData.getRawSentences());
        }

        if (cmd.hasOption("html")) {
            sampler.setupData(trainDebateIndices,
                    debateVoteData.getWords(),
                    debateVoteData.getAuthors(),
                    votes,
                    trainAuthorIndices,
                    trainBillIndices,
                    trainVotes);
            sampler.setTopicVocab(policyAgendaIssues);
            File analysisFolder = new File(samplerFolder, "analysis");
            IOUtils.createFolder(analysisFolder);
            sampler.inputFinalState();
            sampler.outputHTMLFile(new File(samplerFolder, "gop112.html"),
                    trainDebateIndices,
                    debateVoteData.getDocIds(),
                    debateVoteData.getRawSentences(),
                    debateVoteData.getAuthorTable(),
                    GovtrackUrl + congressNum + "/cr/");
        }

        if (cmd.hasOption("testauthor")) {
            File teResultFolder = new File(samplerFolder, TEST_PREFIX + RESULT_FOLDER);
            IOUtils.createFolder(teResultFolder);

            SparseVector[] predictions;
            if (cmd.hasOption("parallel")) {
                predictions = HierMultSHDP.parallelTest(testDebateIndices,
                        debateVoteData.getWords(),
                        debateVoteData.getAuthors(),
                        null,
                        testAuthorIndices,
                        testVotes,
                        new File(samplerFolder, AbstractSampler.IterPredictionFolder),
                        null,
                        sampler);
            } else {
                sampler.setupData(testDebateIndices,
                        debateVoteData.getWords(),
                        debateVoteData.getAuthors(),
                        null,
                        testAuthorIndices,
                        testBillIndices,
                        testVotes);
                predictions = sampler.test();
            }
            AbstractVotePredictor.outputPredictions(new File(teResultFolder, PREDICTION_FILE),
                    votes, predictions);
            AbstractModel.outputPerformances(new File(teResultFolder, RESULT_FILE),
                    AbstractVotePredictor.evaluateAll(votes, testVotes, predictions));
        }
    }

    protected void runRecursiveSLDAIdealPoint(File outputFolder) {
        int K;
        double[][] issuePhis;
        if (cmd.hasOption("K")) {
            issuePhis = null;
            K = Integer.parseInt(cmd.getOptionValue("K"));
        } else {
            issuePhis = estimateIssues();
            K = issuePhis.length;
        }
        double alpha = CLIUtils.getDoubleArgument(cmd, "alpha", 0.1);
        double beta = CLIUtils.getDoubleArgument(cmd, "beta", 0.1);

        double rho = CLIUtils.getDoubleArgument(cmd, "rho", 0.1);
        double sigma = CLIUtils.getDoubleArgument(cmd, "sigma", 0.5);

        double mu = CLIUtils.getDoubleArgument(cmd, "mu", 0.0);
        double gamma = CLIUtils.getDoubleArgument(cmd, "gamma", 2.5);

        RecursiveSLDAIdealPoint sampler = new RecursiveSLDAIdealPoint();
        sampler.setVerbose(verbose);
        sampler.setDebug(debug);
        sampler.setLog(true);
        sampler.setReport(true);
        sampler.setWordVocab(debateVoteData.getWordVocab());
        sampler.setAuthorVocab(debateVoteData.getAuthorVocab());
        sampler.setVoteVocab(debateVoteData.getVoteVocab());

        sampler.configure(outputFolder.getAbsolutePath(),
                debateVoteData.getWordVocab().size(), K,
                alpha, beta, rho, sigma, mu, gamma,
                initState, paramOpt,
                burn_in, max_iters, sample_lag, report_interval);
        File samplerFolder = new File(outputFolder, sampler.getSamplerFolder());
        IOUtils.createFolder(samplerFolder);

        if (cmd.hasOption("train")) {
            sampler.setupData(trainDebateIndices,
                    debateVoteData.getWords(),
                    debateVoteData.getAuthors(),
                    votes,
                    trainAuthorIndices,
                    trainBillIndices,
                    trainVotes);
            sampler.setPriorTopics(issuePhis);
            sampler.setTopicVocab(billData.getTopicVocab());
            sampler.initialize();
            sampler.iterate();

            outputAuthorScore(new File(samplerFolder, AuthorScoreFile),
                    debateVoteData.getAuthorVocab(),
                    trainAuthorIndices,
                    trainVotes,
                    sampler.getUs(),
                    debateVoteData.getAuthorTable());
            outputVoteScores(new File(samplerFolder, VoteScoreFile),
                    debateVoteData.getVoteVocab(),
                    trainBillIndices,
                    sampler.getXs(), sampler.getYs(),
                    debateVoteData.getVoteTable());

            File trResultFolder = new File(samplerFolder, TRAIN_PREFIX + RESULT_FOLDER);
            IOUtils.createFolder(trResultFolder);
            SparseVector[] predictions = sampler.predictInMatrix();
            AbstractVotePredictor.outputPredictions(new File(trResultFolder, PREDICTION_FILE),
                    votes, predictions);
            AbstractModel.outputPerformances(new File(trResultFolder, RESULT_FILE),
                    AbstractVotePredictor.evaluateAll(votes, trainVotes, predictions));
        }

        if (cmd.hasOption("analyzeerror")) {
            analyzeError(samplerFolder);
        }
    }

    protected void runLexicalSLDAIdealPoint(File outputFolder) {
        int K;
        double[][] issuePhis;
        if (cmd.hasOption("K")) {
            issuePhis = null;
            K = Integer.parseInt(cmd.getOptionValue("K"));
        } else {
            issuePhis = estimateIssues();
            K = issuePhis.length;
        }
        double alpha = CLIUtils.getDoubleArgument(cmd, "alpha", 0.1);
        double beta = CLIUtils.getDoubleArgument(cmd, "beta", 0.1);
        double sigma = CLIUtils.getDoubleArgument(cmd, "sigma", 2.5);
        double lambda = CLIUtils.getDoubleArgument(cmd, "lambda", 2.5);
        double l1 = CLIUtils.getDoubleArgument(cmd, "l1", 0.0);
        double l2 = CLIUtils.getDoubleArgument(cmd, "l2", 0.001);
        double gamma = CLIUtils.getDoubleArgument(cmd, "gamma", 2.5);
        double rho = CLIUtils.getDoubleArgument(cmd, "rho", 0.1);

        LexicalSLDAIdealPoint sampler = new LexicalSLDAIdealPoint();
        sampler.setVerbose(verbose);
        sampler.setDebug(debug);
        sampler.setLog(true);
        sampler.setReport(true);
        sampler.setWordVocab(debateVoteData.getWordVocab());
        sampler.setAuthorVocab(debateVoteData.getAuthorVocab());
        sampler.setVoteVocab(debateVoteData.getVoteVocab());

        if (lambda != 0) {
            sampler.configure(outputFolder.getAbsolutePath(),
                    debateVoteData.getWordVocab().size(), K,
                    alpha, beta, rho, sigma, gamma, lambda, 0.0, 0.0, issuePhis,
                    initState, paramOpt,
                    burn_in, max_iters, sample_lag, report_interval);
        } else {
            sampler.configure(outputFolder.getAbsolutePath(),
                    debateVoteData.getWordVocab().size(), K,
                    alpha, beta, rho, sigma, gamma, 0.0, l1, l2, issuePhis,
                    initState, paramOpt,
                    burn_in, max_iters, sample_lag, report_interval);
        }
        File samplerFolder = new File(outputFolder, sampler.getSamplerFolder());
        IOUtils.createFolder(samplerFolder);
        File trResultFolder = new File(samplerFolder, TRAIN_PREFIX + RESULT_FOLDER);
        IOUtils.createFolder(trResultFolder);

        if (cmd.hasOption("train")) {
            sampler.setupData(trainDebateIndices,
                    debateVoteData.getWords(),
                    debateVoteData.getAuthors(),
                    votes,
                    trainAuthorIndices,
                    trainBillIndices,
                    trainVotes);
            sampler.initialize();
            sampler.metaIterate();
            if (issuePhis == null) {
                sampler.outputTopicTopWords(new File(samplerFolder, TopWordFile),
                        numTopWords);
            } else {
                sampler.outputTopicTopWords(new File(samplerFolder, TopWordFile),
                        numTopWords, billData.getTopicVocab());
            }
            sampler.outputLexicalItems(new File(samplerFolder, LexicalFile));

            outputAuthorScore(new File(samplerFolder, AuthorScoreFile),
                    debateVoteData.getAuthorVocab(),
                    trainAuthorIndices,
                    trainVotes,
                    sampler.getUs(),
                    debateVoteData.getAuthorTable());
            outputVoteScores(new File(samplerFolder, VoteScoreFile),
                    debateVoteData.getVoteVocab(),
                    trainBillIndices,
                    sampler.getXs(), sampler.getYs(),
                    debateVoteData.getVoteTable());

            SparseVector[] predictions = sampler.predictInMatrix();
            AbstractVotePredictor.outputPredictions(new File(trResultFolder, PREDICTION_FILE),
                    votes, predictions);
            AbstractModel.outputPerformances(new File(trResultFolder, RESULT_FILE),
                    AbstractVotePredictor.evaluateAll(votes, trainVotes, predictions));
        }

        if (cmd.hasOption("coherence")) {
            sampler.computeAvgTopicCoherence(new File(trResultFolder, TopicCoherenceFile),
                    debateVoteData.getTopicCoherence());
        }

        if (cmd.hasOption("testvote")) { // average over multiple test chains
            sampler.setupData(testDebateIndices,
                    debateVoteData.getWords(),
                    debateVoteData.getAuthors(),
                    null,
                    testAuthorIndices,
                    testBillIndices,
                    testVotes);
            sampler.inputFinalState();
            SparseVector[] predictions = sampler.predictInMatrixMultiples();

            File teResultFolder = new File(samplerFolder, TEST_PREFIX + RESULT_FOLDER);
            IOUtils.createFolder(teResultFolder);
            AbstractVotePredictor.outputPredictions(new File(teResultFolder, PREDICTION_FILE),
                    votes, predictions);
            AbstractModel.outputPerformances(new File(teResultFolder, RESULT_FILE),
                    AbstractVotePredictor.evaluateAll(votes, testVotes, predictions));
        }

        if (cmd.hasOption("testauthor")) {
            File teResultFolder = new File(samplerFolder, TEST_PREFIX + RESULT_FOLDER);
            IOUtils.createFolder(teResultFolder);

            SparseVector[] predictions;
            if (cmd.hasOption("parallel")) {
                predictions = LexicalSLDAIdealPoint.parallelTest(testDebateIndices,
                        debateVoteData.getWords(),
                        debateVoteData.getAuthors(),
                        testAuthorIndices,
                        testVotes,
                        new File(samplerFolder, AbstractSampler.IterPredictionFolder),
                        sampler);
            } else {
                sampler.setupData(testDebateIndices,
                        debateVoteData.getWords(),
                        debateVoteData.getAuthors(),
                        null,
                        testAuthorIndices,
                        testBillIndices,
                        testVotes);
                predictions = sampler.test(null, null,
                        new File(samplerFolder, TEST_PREFIX + AssignmentFile));
                outputAuthorScore(new File(teResultFolder, AuthorScoreFile),
                        debateVoteData.getAuthorVocab(),
                        testAuthorIndices,
                        testVotes,
                        sampler.getPredictedUs(),
                        debateVoteData.getAuthorTable());
            }
            AbstractVotePredictor.outputPredictions(new File(teResultFolder, PREDICTION_FILE),
                    votes, predictions);
            AbstractModel.outputPerformances(new File(teResultFolder, RESULT_FILE),
                    AbstractVotePredictor.evaluateAll(votes, testVotes, predictions));
        }

        if (cmd.hasOption("analyzeerror")) {
            analyzeError(samplerFolder);
        }
    }

    protected void runHybridSLDAIdealPoint(File outputFolder) {
        int K;
        double[][] issuePhis;
        if (cmd.hasOption("K")) {
            issuePhis = null;
            K = Integer.parseInt(cmd.getOptionValue("K"));
        } else {
            issuePhis = estimateIssues();
            K = issuePhis.length;
        }
        double alpha = CLIUtils.getDoubleArgument(cmd, "alpha", 0.1);
        double beta = CLIUtils.getDoubleArgument(cmd, "beta", 0.1);
        double rho = CLIUtils.getDoubleArgument(cmd, "rho", 0.1);
        double sigma = CLIUtils.getDoubleArgument(cmd, "sigma", 0.5);
        double gamma = CLIUtils.getDoubleArgument(cmd, "gamma", 5.0);
        double lambda = CLIUtils.getDoubleArgument(cmd, "lambda", 5);
        double l1 = CLIUtils.getDoubleArgument(cmd, "l1", 0.0);
        double l2 = CLIUtils.getDoubleArgument(cmd, "l2", 0.001);

        HybridSLDAIdealPoint sampler = new HybridSLDAIdealPoint();
        sampler.setVerbose(verbose);
        sampler.setDebug(debug);
        sampler.setLog(true);
        sampler.setReport(true);
        sampler.setWordVocab(debateVoteData.getWordVocab());
        sampler.setAuthorVocab(debateVoteData.getAuthorVocab());
        sampler.setVoteVocab(debateVoteData.getVoteVocab());
        if (!cmd.hasOption("K")) {
            sampler.setLabelVocab(billData.getTopicVocab());
        }

        sampler.configure(outputFolder.getAbsolutePath(),
                debateVoteData.getWordVocab().size(), K,
                alpha, beta, rho, sigma, gamma, lambda, l1, l2, issuePhis,
                initState, paramOpt,
                burn_in, max_iters, sample_lag, report_interval);

        File samplerFolder = new File(outputFolder, sampler.getSamplerFolder());
        IOUtils.createFolder(samplerFolder);
        File trResultFolder = new File(samplerFolder, TRAIN_PREFIX + RESULT_FOLDER);
        IOUtils.createFolder(trResultFolder);
        File teResultFolder = new File(samplerFolder, TEST_PREFIX + RESULT_FOLDER);

        if (cmd.hasOption("train")) {
            sampler.setupData(trainDebateIndices,
                    debateVoteData.getWords(),
                    debateVoteData.getAuthors(),
                    votes,
                    trainAuthorIndices,
                    trainBillIndices,
                    trainVotes);
            sampler.initialize();
            sampler.metaIterate();
            sampler.outputTopicTopWords(new File(samplerFolder, TopWordFile),
                    numTopWords);

            outputAuthorScore(new File(samplerFolder, AuthorScoreFile),
                    debateVoteData.getAuthorVocab(),
                    trainAuthorIndices,
                    trainVotes,
                    sampler.getUs(),
                    debateVoteData.getAuthorTable());
            outputVoteScores(new File(samplerFolder, VoteScoreFile),
                    debateVoteData.getVoteVocab(),
                    trainBillIndices,
                    sampler.getXs(), sampler.getYs(),
                    debateVoteData.getVoteTable());
            SparseVector[] predictions = sampler.predictInMatrix();
            AbstractVotePredictor.outputPredictions(new File(trResultFolder, PREDICTION_FILE),
                    votes, predictions);
            AbstractModel.outputPerformances(new File(trResultFolder, RESULT_FILE),
                    AbstractVotePredictor.evaluateAll(votes, trainVotes, predictions));
        }

        if (cmd.hasOption("coherence")) {
            sampler.computeAvgTopicCoherence(new File(trResultFolder, TopicCoherenceFile),
                    debateVoteData.getTopicCoherence());
        }

        if (cmd.hasOption("testvote")) { // average over multiple test chains
            IOUtils.createFolder(teResultFolder);

            sampler.setupData(testDebateIndices,
                    debateVoteData.getWords(),
                    debateVoteData.getAuthors(),
                    null,
                    testAuthorIndices,
                    testBillIndices,
                    testVotes);
            sampler.inputFinalState();
            SparseVector[] predictions = sampler.predictInMatrixMultiples();

            AbstractVotePredictor.outputPredictions(new File(teResultFolder, PREDICTION_FILE),
                    votes, predictions);
            AbstractModel.outputPerformances(new File(teResultFolder, RESULT_FILE),
                    AbstractVotePredictor.evaluateAll(votes, testVotes, predictions));
        }

        if (cmd.hasOption("testauthor")) {
            IOUtils.createFolder(teResultFolder);

            SparseVector[] predictions;
            if (cmd.hasOption("parallel")) {
                predictions = HybridSLDAIdealPoint.parallelTest(testDebateIndices,
                        debateVoteData.getWords(),
                        debateVoteData.getAuthors(),
                        testAuthorIndices,
                        testVotes,
                        new File(samplerFolder, AbstractSampler.IterPredictionFolder),
                        sampler);
            } else {
                sampler.setupData(testDebateIndices,
                        debateVoteData.getWords(),
                        debateVoteData.getAuthors(),
                        votes,
                        testAuthorIndices,
                        testBillIndices,
                        testVotes);
                predictions = sampler.test(null, null,
                        new File(samplerFolder, TEST_PREFIX + AssignmentFile));
                outputAuthorScore(new File(teResultFolder, AuthorScoreFile),
                        debateVoteData.getAuthorVocab(),
                        testAuthorIndices,
                        testVotes,
                        sampler.getPredictedUs(),
                        debateVoteData.getAuthorTable());
            }
            AbstractVotePredictor.outputPredictions(new File(teResultFolder, PREDICTION_FILE),
                    votes, predictions);
            AbstractModel.outputPerformances(new File(teResultFolder, RESULT_FILE),
                    AbstractVotePredictor.evaluateAll(votes, testVotes, predictions));
        }

        if (cmd.hasOption("analyzeerror")) {
            analyzeError(samplerFolder);
        }

        if (cmd.hasOption("dev")) {
            File devFolder = new File(outputFolder, sampler.getBasename());
            IOUtils.createFolder(devFolder);
            try {
                File devFile = new File(devFolder, "dev.txt");
                BufferedWriter writer = IOUtils.getBufferedWriter(devFile);
                boolean hasHeader = false;
                lambda = 0.0;
                double[] l1s = {0.001, 0.0001, 0.00001, 0.000001};
                double[] l2s = {0.001, 0.0001, 0.00001, 0.000001};

                for (double ll1 : l1s) {
                    for (double ll2 : l2s) {
                        if (verbose) {
                            logln("Tuning: l1 = " + ll1 + ". l2 = " + ll2);
                        }

                        sampler.setPrefix("");
                        sampler.configure(devFolder.getAbsolutePath(),
                                debateVoteData.getWordVocab().size(), K,
                                alpha, beta, rho, sigma, gamma, lambda, ll1, ll2, issuePhis,
                                initState, paramOpt,
                                burn_in, max_iters, sample_lag, report_interval);

                        samplerFolder = new File(devFolder, sampler.getSamplerFolder());
                        trResultFolder = new File(samplerFolder, TRAIN_PREFIX + RESULT_FOLDER);
                        IOUtils.createFolder(trResultFolder);
                        teResultFolder = new File(samplerFolder, TEST_PREFIX + RESULT_FOLDER);
                        IOUtils.createFolder(teResultFolder);

                        sampler.setupData(trainDebateIndices,
                                debateVoteData.getWords(),
                                debateVoteData.getAuthors(),
                                votes,
                                trainAuthorIndices,
                                trainBillIndices,
                                trainVotes);
                        sampler.initialize();
                        sampler.metaIterate();

                        SparseVector[] predictions = sampler.predictInMatrix();
                        AbstractVotePredictor.outputPredictions(new File(trResultFolder, PREDICTION_FILE),
                                votes, predictions);
                        AbstractModel.outputPerformances(new File(trResultFolder, RESULT_FILE),
                                AbstractVotePredictor.evaluateAll(votes, trainVotes, predictions));

//                        sampler.computeAvgTopicCoherence(new File(trResultFolder, TopicCoherenceFile),
//                                debateVoteData.getTopicCoherence());
                        predictions = HybridSLDAIdealPoint.parallelTest(testDebateIndices,
                                debateVoteData.getWords(),
                                debateVoteData.getAuthors(),
                                testAuthorIndices,
                                testVotes,
                                new File(samplerFolder, AbstractSampler.IterPredictionFolder),
                                sampler);
                        AbstractVotePredictor.outputPredictions(new File(teResultFolder, PREDICTION_FILE),
                                votes, predictions);
                        ArrayList<Measurement> results = AbstractVotePredictor.evaluateAll(votes, testVotes, predictions);
                        AbstractModel.outputPerformances(new File(teResultFolder, RESULT_FILE), results);

                        if (!hasHeader) {
                            writer.write("l1\tl2");
                            for (Measurement m : results) {
                                writer.write("\t" + m.getName());
                            }
                            writer.write("\n");
                            hasHeader = true;
                        }

                        writer.write(ll1 + "\t" + ll2);
                        for (Measurement m : results) {
                            writer.write("\t" + m.getValue());
                        }
                        writer.write("\n");
                    }
                }
                writer.close();

                logln("Written to " + devFile);
            } catch (Exception e) {
                e.printStackTrace();
                throw new RuntimeException("Exception while tuning");
            }
        }
    }

    protected void runHybridSLDAMultipleIdealPoint(File outputFolder) {
        int K;
        double[][] issuePhis;
        if (cmd.hasOption("K")) {
            issuePhis = null;
            K = Integer.parseInt(cmd.getOptionValue("K"));
        } else {
            issuePhis = estimateIssues();
            K = issuePhis.length;
        }
        double alpha = CLIUtils.getDoubleArgument(cmd, "alpha", 0.1);
        double beta = CLIUtils.getDoubleArgument(cmd, "beta", 0.1);
        double rho = CLIUtils.getDoubleArgument(cmd, "rho", 0.1);
        double sigma = CLIUtils.getDoubleArgument(cmd, "sigma", 0.1);
        double gamma = CLIUtils.getDoubleArgument(cmd, "gamma", 5.0);
        double l1 = CLIUtils.getDoubleArgument(cmd, "l1", 0.0);
        double l2 = CLIUtils.getDoubleArgument(cmd, "l2", 0.001);

        HybridSLDAMultipleIdealPoint sampler = new HybridSLDAMultipleIdealPoint();
        sampler.setVerbose(verbose);
        sampler.setDebug(debug);
        sampler.setLog(true);
        sampler.setReport(true);
        sampler.setWordVocab(debateVoteData.getWordVocab());
        sampler.setAuthorVocab(debateVoteData.getAuthorVocab());
        sampler.setVoteVocab(debateVoteData.getVoteVocab());
        if (!cmd.hasOption("K")) {
            sampler.setLabelVocab(billData.getTopicVocab());
        }

        sampler.configure(outputFolder.getAbsolutePath(),
                debateVoteData.getWordVocab().size(), K,
                alpha, beta, rho, sigma, gamma, l1, l2, issuePhis,
                initState, paramOpt,
                burn_in, max_iters, sample_lag, report_interval);

        File samplerFolder = new File(outputFolder, sampler.getSamplerFolder());
        IOUtils.createFolder(samplerFolder);
        File trResultFolder = new File(samplerFolder, TRAIN_PREFIX + RESULT_FOLDER);
        IOUtils.createFolder(trResultFolder);
        File teResultFolder = new File(samplerFolder, TEST_PREFIX + RESULT_FOLDER);

        if (cmd.hasOption("train")) {
            sampler.setupData(trainDebateIndices,
                    debateVoteData.getWords(),
                    debateVoteData.getAuthors(),
                    votes,
                    trainAuthorIndices,
                    trainBillIndices,
                    trainVotes);
            sampler.initialize();
            sampler.metaIterate();
            sampler.outputTopicTopWords(new File(samplerFolder, TopWordFile),
                    numTopWords);
            SparseVector[] predictions = sampler.predictInMatrix();
            AbstractVotePredictor.outputPredictions(new File(trResultFolder, PREDICTION_FILE),
                    votes, predictions);
            AbstractModel.outputPerformances(new File(trResultFolder, RESULT_FILE),
                    AbstractVotePredictor.evaluateAll(votes, trainVotes, predictions));
        }

//        if (cmd.hasOption("testauthor")) {
//            IOUtils.createFolder(teResultFolder);
//
//            SparseVector[] predictions;
//            if (cmd.hasOption("parallel")) {
//                predictions = HybridSLDAIdealPoint.parallelTest(testDebateIndices,
//                        debateVoteData.getWords(),
//                        debateVoteData.getAuthors(),
//                        testAuthorIndices,
//                        testVotes,
//                        new File(samplerFolder, AbstractSampler.IterPredictionFolder),
//                        sampler);
//            } else {
//                sampler.setupData(testDebateIndices,
//                        debateVoteData.getWords(),
//                        debateVoteData.getAuthors(),
//                        null,
//                        testAuthorIndices,
//                        testBillIndices,
//                        testVotes);
//                predictions = sampler.test(null, null,
//                        new File(samplerFolder, TEST_PREFIX + AssignmentFile));
//                outputAuthorScore(new File(teResultFolder, AuthorScoreFile),
//                        debateVoteData.getAuthorVocab(),
//                        testAuthorIndices,
//                        testVotes,
//                        sampler.getPredictedUs(),
//                        debateVoteData.getAuthorTable());
//            }
//            AbstractVotePredictor.outputPredictions(new File(teResultFolder, PREDICTION_FILE),
//                    votes, predictions);
//            AbstractModel.outputPerformances(new File(teResultFolder, RESULT_FILE),
//                    AbstractVotePredictor.evaluateAll(votes, testVotes, predictions));
//        }
    }

    protected void runSLDAMultIdealPoint(File outputFolder) {
        int K;
        double[][] issuePhis;
        if (cmd.hasOption("K")) {
            issuePhis = null;
            K = Integer.parseInt(cmd.getOptionValue("K"));
        } else {
            issuePhis = estimateIssues();
            K = issuePhis.length;
        }
        double alpha = CLIUtils.getDoubleArgument(cmd, "alpha", 0.1);
        double beta = CLIUtils.getDoubleArgument(cmd, "beta", 0.1);
        double rho = CLIUtils.getDoubleArgument(cmd, "rho", 2.5);
        double sigma = CLIUtils.getDoubleArgument(cmd, "sigma", 2.5);
        double gamma = CLIUtils.getDoubleArgument(cmd, "gamma", 2.5);
        double l1 = CLIUtils.getDoubleArgument(cmd, "l1", 0.0);
        double l2 = CLIUtils.getDoubleArgument(cmd, "l2", 2.5);

        SLDAMultIdealPoint sampler = new SLDAMultIdealPoint();
        sampler.setVerbose(verbose);
        sampler.setDebug(debug);
        sampler.setLog(true);
        sampler.setReport(true);
        sampler.setWordVocab(debateVoteData.getWordVocab());
        sampler.setAuthorVocab(debateVoteData.getAuthorVocab());
        sampler.setVoteVocab(debateVoteData.getVoteVocab());

        sampler.configure(outputFolder.getAbsolutePath(),
                debateVoteData.getWordVocab().size(), K,
                alpha, beta, rho, sigma, gamma, l1, l2,
                initState, paramOpt,
                burn_in, max_iters, sample_lag, report_interval);
        File samplerFolder = new File(outputFolder, sampler.getSamplerFolder());
        IOUtils.createFolder(samplerFolder);

        File teResultFolder = new File(samplerFolder, TEST_PREFIX + RESULT_FOLDER);
        File trResultFolder = new File(samplerFolder, TRAIN_PREFIX + RESULT_FOLDER);

        if (cmd.hasOption("train")) {
            sampler.setupData(trainDebateIndices,
                    debateVoteData.getWords(),
                    debateVoteData.getAuthors(),
                    votes,
                    trainAuthorIndices,
                    trainBillIndices,
                    trainVotes);
            sampler.initialize(issuePhis);
            sampler.iterate();
            if (issuePhis == null) {
                sampler.outputTopicTopWords(new File(samplerFolder, TopWordFile),
                        numTopWords);
            } else {
                sampler.outputTopicTopWords(new File(samplerFolder, TopWordFile),
                        numTopWords, billData.getTopicVocab());
            }
            sampler.outputAuthorIdealPoints(new File(samplerFolder, AuthorScoreFile),
                    sampler.getUs(), null);
            sampler.outputVoteIdealPoints(new File(samplerFolder, VoteScoreFile),
                    sampler.getXs(), null);

            // training predictions
            IOUtils.createFolder(trResultFolder);
            SparseVector[] predictions = sampler.predictInMatrix();
            AbstractVotePredictor.outputPredictions(new File(trResultFolder, PREDICTION_FILE),
                    votes, predictions);
            AbstractModel.outputPerformances(new File(trResultFolder, RESULT_FILE),
                    AbstractVotePredictor.evaluateAll(votes, trainVotes, predictions));
        }

        if (cmd.hasOption("testvote")) { // average over multiple test chains
            IOUtils.createFolder(teResultFolder);
            sampler.setupData(testDebateIndices,
                    debateVoteData.getWords(),
                    debateVoteData.getAuthors(),
                    null,
                    testAuthorIndices,
                    testBillIndices,
                    testVotes);
            SparseVector[] predictions = sampler.predictInMatrixMultiples();
            AbstractVotePredictor.outputPredictions(new File(teResultFolder, PREDICTION_FILE),
                    votes, predictions);
            AbstractModel.outputPerformances(new File(teResultFolder, RESULT_FILE),
                    AbstractVotePredictor.evaluateAll(votes, testVotes, predictions));
        }

        if (cmd.hasOption("testauthor")) {
            IOUtils.createFolder(teResultFolder);
            SparseVector[] predictions;
            if (cmd.hasOption("parallel")) {
                predictions = SLDAMultIdealPoint.parallelTest(testDebateIndices,
                        debateVoteData.getWords(),
                        debateVoteData.getAuthors(),
                        testAuthorIndices,
                        testVotes,
                        new File(samplerFolder, AbstractSampler.IterPredictionFolder),
                        sampler);
            } else {
                sampler.setupData(testDebateIndices,
                        debateVoteData.getWords(),
                        debateVoteData.getAuthors(),
                        null,
                        testAuthorIndices,
                        testBillIndices,
                        testVotes);
                predictions = sampler.test(null, null,
                        new File(samplerFolder, TEST_PREFIX + AssignmentFile));
                sampler.outputAuthorIdealPoints(new File(teResultFolder, AuthorScoreFile),
                        sampler.getPredictedUs(), null);
                sampler.outputVoteIdealPoints(new File(teResultFolder, VoteScoreFile),
                        sampler.getXs(), null);
            }
            AbstractVotePredictor.outputPredictions(new File(teResultFolder, PREDICTION_FILE),
                    votes, predictions);
            AbstractModel.outputPerformances(new File(teResultFolder, RESULT_FILE),
                    AbstractVotePredictor.evaluateAll(votes, testVotes, predictions));
        }

        if (cmd.hasOption("analyzeerror")) {
            analyzeError(samplerFolder);
        }
    }

    protected void runSNLDAIdealPoint(File outputFolder) {
        // estimate seeded issues
        double[][] issuePhis;
        if (cmd.hasOption("K")) {
            int V = debateVoteData.getWordVocab().size();
            int K = Integer.parseInt(cmd.getOptionValue("K"));
            issuePhis = new double[K][V];
            for (int kk = 0; kk < K; kk++) {
                Arrays.fill(issuePhis[kk], 1.0 / V);
            }
        } else {
            issuePhis = estimateIssues();
        }
        int J = CLIUtils.getIntegerArgument(cmd, "J", 3);
        double[] alphas = CLIUtils.getDoubleArrayArgument(cmd, "alphas",
                new double[]{0.1, 0.1}, ",");
        double[] betas = CLIUtils.getDoubleArrayArgument(cmd, "betas",
                new double[]{10, 10, 0.1}, ",");
        double[] gamma_means = CLIUtils.getDoubleArrayArgument(cmd, "gamma-means",
                new double[]{0.2, 0.2}, ",");
        double[] gamma_scales = CLIUtils.getDoubleArrayArgument(cmd, "gamma-scales",
                new double[]{10, 1}, ",");
        double mu = CLIUtils.getDoubleArgument(cmd, "mu", 0.0);
        double sigma = CLIUtils.getDoubleArgument(cmd, "sigma", 10);
        double rho = CLIUtils.getDoubleArgument(cmd, "rho", 1.0);
        boolean hasRootTopic = cmd.hasOption("roottopic");

        SNLDAIdealPoint sampler = new SNLDAIdealPoint();
        sampler.setVerbose(verbose);
        sampler.setDebug(debug);
        sampler.setLog(true);
        sampler.setReport(true);
        sampler.setWordVocab(debateVoteData.getWordVocab());
        sampler.setAuthorVocab(debateVoteData.getAuthorVocab());
        if (!cmd.hasOption("K")) {
            sampler.setLabelVocab(billData.getTopicVocab());
        }

        PathAssumption pathAssumption = PathAssumption.MAXIMAL;
        String path = CLIUtils.getStringArgument(cmd, "path", "max");
        switch (path) {
            case "max":
                pathAssumption = PathAssumption.MAXIMAL;
                break;
            case "min":
                pathAssumption = PathAssumption.MINIMAL;
                break;
            default:
                throw new RuntimeException("Path assumption " + path + " not supported");
        }

        sampler.configure(outputFolder.getAbsolutePath(),
                debateVoteData.getWordVocab().size(), J,
                issuePhis, alphas, betas, gamma_means, gamma_scales,
                rho, mu, sigma, hasRootTopic,
                initState, pathAssumption, paramOpt,
                burn_in, max_iters, sample_lag, report_interval);
        File samplerFolder = new File(outputFolder, sampler.getSamplerFolder());
        IOUtils.createFolder(samplerFolder);

        File teResultFolder = new File(samplerFolder, TEST_PREFIX + RESULT_FOLDER);
        File trResultFolder = new File(samplerFolder, TRAIN_PREFIX + RESULT_FOLDER);

        if (cmd.hasOption("train")) {
            sampler.setupData(trainDebateIndices,
                    debateVoteData.getWords(),
                    debateVoteData.getAuthors(),
                    votes, trainAuthorIndices, trainBillIndices,
                    trainVotes);
            sampler.initialize();
            sampler.iterate();
            sampler.outputTopicTopWords(new File(samplerFolder, TopWordFile), numTopWords);
            outputAuthorScore(new File(samplerFolder, AuthorScoreFile),
                    debateVoteData.getAuthorVocab(),
                    trainAuthorIndices,
                    trainVotes,
                    sampler.getUs(),
                    debateVoteData.getAuthorTable());
            outputVoteScores(new File(samplerFolder, VoteScoreFile),
                    debateVoteData.getVoteVocab(), this.trainBillIndices,
                    sampler.getXs(), sampler.getYs(),
                    debateVoteData.getVoteTable());

            SparseVector[] predictions = sampler.predictInMatrix();
            IOUtils.createFolder(trResultFolder);
            AbstractVotePredictor.outputPredictions(new File(trResultFolder, PREDICTION_FILE),
                    votes, predictions);
            AbstractModel.outputPerformances(new File(trResultFolder, RESULT_FILE),
                    AbstractVotePredictor.evaluateAll(votes, trainVotes, predictions));
        }

        if (cmd.hasOption("testvote")) { // average over multiple test chains
            IOUtils.createFolder(teResultFolder);
            sampler.setupData(testDebateIndices,
                    debateVoteData.getWords(),
                    debateVoteData.getAuthors(),
                    null,
                    testAuthorIndices,
                    testBillIndices,
                    testVotes);
            SparseVector[] predictions = sampler.predictInMatrixMultiples();
            AbstractVotePredictor.outputPredictions(new File(teResultFolder, PREDICTION_FILE),
                    votes, predictions);
            AbstractModel.outputPerformances(new File(teResultFolder, RESULT_FILE),
                    AbstractVotePredictor.evaluateAll(votes, testVotes, predictions));
        }

        if (cmd.hasOption("testauthor")) {
            IOUtils.createFolder(teResultFolder);
            SparseVector[] predictions;
            if (cmd.hasOption("parallel")) {
                File iterPredFolderPath = new File(samplerFolder,
                        AbstractSampler.IterPredictionFolder);
                predictions = SNLDAIdealPoint.parallelTest(testDebateIndices,
                        debateVoteData.getWords(),
                        debateVoteData.getAuthors(),
                        testAuthorIndices,
                        testVotes,
                        iterPredFolderPath,
                        sampler);

            } else {
                sampler.setupData(testDebateIndices,
                        debateVoteData.getWords(),
                        debateVoteData.getAuthors(),
                        null,
                        testAuthorIndices,
                        testBillIndices,
                        testVotes);
                predictions = sampler.test(null, null,
                        new File(samplerFolder, TEST_PREFIX + AssignmentFile));
                outputAuthorScore(new File(teResultFolder, AuthorScoreFile),
                        debateVoteData.getAuthorVocab(),
                        testAuthorIndices,
                        testVotes,
                        sampler.getPredictedUs(),
                        debateVoteData.getAuthorTable());
            }
            AbstractVotePredictor.outputPredictions(new File(teResultFolder, PREDICTION_FILE),
                    votes, predictions);
            AbstractModel.outputPerformances(new File(teResultFolder, RESULT_FILE),
                    AbstractVotePredictor.evaluateAll(votes, testVotes, predictions));
        }

        if (cmd.hasOption("analyzeerror")) {
            analyzeError(samplerFolder);
        }

        if (cmd.hasOption("visualize")) {
            sampler.setupData(trainDebateIndices,
                    debateVoteData.getWords(),
                    debateVoteData.getAuthors(),
                    votes,
                    trainAuthorIndices,
                    trainBillIndices,
                    trainVotes);
            sampler.inputFinalState();
            File htmlFile = new File(samplerFolder, sampler.getBasename()
                    + "-" + congressNum + ".html");
            sampler.outputHTML(htmlFile,
                    trainDebateIndices,
                    debateVoteData.getDocIds(),
                    debateVoteData.getRawSentences(),
                    debateVoteData.getAuthorTable(),
                    GovtrackUrl + congressNum + "/cr/");
        }

        if (cmd.hasOption("testeval")) {
            SparseVector[] predictions = AbstractVotePredictor.inputPredictions(
                    new File(teResultFolder, PREDICTION_FILE));
            AbstractModel.outputPerformances(new File(teResultFolder, RESULT_FILE),
                    AbstractVotePredictor.evaluateAll(votes, testVotes, predictions));
        }
    }

    protected void runLexicalSNLDAIdealPoint(File outputFolder) {
        // estimate seeded issues
        double[][] issuePhis;
        if (cmd.hasOption("K")) {
            int V = debateVoteData.getWordVocab().size();
            int K = Integer.parseInt(cmd.getOptionValue("K"));
            issuePhis = new double[K][V];
            for (int kk = 0; kk < K; kk++) {
                Arrays.fill(issuePhis[kk], 1.0 / V);
            }
        } else {
            issuePhis = estimateIssues();
        }
        int J = CLIUtils.getIntegerArgument(cmd, "J", 3);
        double[] alphas = CLIUtils.getDoubleArrayArgument(cmd, "alphas",
                new double[]{0.1, 0.1}, ",");
        double[] betas = CLIUtils.getDoubleArrayArgument(cmd, "betas",
                new double[]{10, 10, 0.1}, ",");
        double[] gamma_means = CLIUtils.getDoubleArrayArgument(cmd, "gamma-means",
                new double[]{0.2, 0.2}, ",");
        double[] gamma_scales = CLIUtils.getDoubleArrayArgument(cmd, "gamma-scales",
                new double[]{10, 1}, ",");
        double[] sigmas = CLIUtils.getDoubleArrayArgument(cmd, "sigmas",
                new double[]{0.0, 2.5, 5}, ",");
        double sigma = CLIUtils.getDoubleArgument(cmd, "sigma", 10);
        double rho = CLIUtils.getDoubleArgument(cmd, "rho", 0.05);
        double lambda = CLIUtils.getDoubleArgument(cmd, "lambda", 2.5);
        boolean hasRootTopic = cmd.hasOption("roottopic");

        LexicalSNLDAIdealPoint sampler = new LexicalSNLDAIdealPoint();
        sampler.setVerbose(verbose);
        sampler.setDebug(debug);
        sampler.setLog(true);
        sampler.setReport(true);
        sampler.setWordVocab(debateVoteData.getWordVocab());
        sampler.setAuthorVocab(debateVoteData.getAuthorVocab());
        if (!cmd.hasOption("K")) {
            sampler.setLabelVocab(billData.getTopicVocab());
        }

        PathAssumption pathAssumption = PathAssumption.MAXIMAL;
        String path = CLIUtils.getStringArgument(cmd, "path", "max");
        switch (path) {
            case "max":
                pathAssumption = PathAssumption.MAXIMAL;
                break;
            case "min":
                pathAssumption = PathAssumption.MINIMAL;
                break;
            default:
                throw new RuntimeException("Path assumption " + path + " not supported");
        }

        sampler.configure(outputFolder.getAbsolutePath(),
                debateVoteData.getWordVocab().size(), J,
                issuePhis, alphas, betas, gamma_means, gamma_scales,
                rho, sigmas, sigma, lambda, hasRootTopic,
                initState, pathAssumption, paramOpt,
                burn_in, max_iters, sample_lag, report_interval);
        File samplerFolder = new File(outputFolder, sampler.getSamplerFolder());
        IOUtils.createFolder(samplerFolder);

        File teResultFolder = new File(samplerFolder, TEST_PREFIX + RESULT_FOLDER);
        File trResultFolder = new File(samplerFolder, TRAIN_PREFIX + RESULT_FOLDER);

        if (cmd.hasOption("train")) {
            sampler.setupData(trainDebateIndices,
                    debateVoteData.getWords(),
                    debateVoteData.getAuthors(),
                    votes, trainAuthorIndices, trainBillIndices,
                    trainVotes);
            sampler.initialize();
            sampler.iterate();
            sampler.outputWords(new File(samplerFolder, "lex-words.txt"));
            sampler.outputTopicTopWords(new File(samplerFolder, TopWordFile), numTopWords);
            outputAuthorScore(new File(samplerFolder, AuthorScoreFile),
                    debateVoteData.getAuthorVocab(),
                    trainAuthorIndices,
                    trainVotes,
                    sampler.getUs(),
                    debateVoteData.getAuthorTable());
            outputVoteScores(new File(samplerFolder, VoteScoreFile),
                    debateVoteData.getVoteVocab(), this.trainBillIndices,
                    sampler.getXs(), sampler.getYs(),
                    debateVoteData.getVoteTable());

            SparseVector[] predictions = sampler.predictInMatrix();
            IOUtils.createFolder(trResultFolder);
            AbstractVotePredictor.outputPredictions(new File(trResultFolder, PREDICTION_FILE),
                    votes, predictions);
            AbstractModel.outputPerformances(new File(trResultFolder, RESULT_FILE),
                    AbstractVotePredictor.evaluateAll(votes, trainVotes, predictions));
        }

        if (cmd.hasOption("testvote")) { // average over multiple test chains
            IOUtils.createFolder(teResultFolder);
            sampler.setupData(testDebateIndices,
                    debateVoteData.getWords(),
                    debateVoteData.getAuthors(),
                    null,
                    testAuthorIndices,
                    testBillIndices,
                    testVotes);
            SparseVector[] predictions = sampler.predictInMatrixMultiples();
            AbstractVotePredictor.outputPredictions(new File(teResultFolder, PREDICTION_FILE),
                    votes, predictions);
            AbstractModel.outputPerformances(new File(teResultFolder, RESULT_FILE),
                    AbstractVotePredictor.evaluateAll(votes, testVotes, predictions));
        }

        if (cmd.hasOption("testauthor")) {
            IOUtils.createFolder(teResultFolder);
            SparseVector[] predictions;
            if (cmd.hasOption("parallel")) {
                File iterPredFolderPath = new File(samplerFolder,
                        AbstractSampler.IterPredictionFolder);
                predictions = LexicalSNLDAIdealPoint.parallelTest(testDebateIndices,
                        debateVoteData.getWords(),
                        debateVoteData.getAuthors(),
                        testAuthorIndices,
                        testVotes,
                        iterPredFolderPath,
                        sampler);

            } else {
                sampler.setupData(testDebateIndices,
                        debateVoteData.getWords(),
                        debateVoteData.getAuthors(),
                        null,
                        testAuthorIndices,
                        testBillIndices,
                        testVotes);
                predictions = sampler.test(null, null,
                        new File(samplerFolder, TEST_PREFIX + AssignmentFile));
                outputAuthorScore(new File(teResultFolder, AuthorScoreFile),
                        debateVoteData.getAuthorVocab(),
                        testAuthorIndices,
                        testVotes,
                        sampler.getPredictedUs(),
                        debateVoteData.getAuthorTable());
            }
            AbstractVotePredictor.outputPredictions(new File(teResultFolder, PREDICTION_FILE),
                    votes, predictions);
            AbstractModel.outputPerformances(new File(teResultFolder, RESULT_FILE),
                    AbstractVotePredictor.evaluateAll(votes, testVotes, predictions));
        }

        if (cmd.hasOption("visualize")) {
            sampler.setupData(trainDebateIndices,
                    debateVoteData.getWords(),
                    debateVoteData.getAuthors(),
                    votes,
                    trainAuthorIndices,
                    trainBillIndices,
                    trainVotes);
            sampler.inputFinalState();
            File htmlFile = new File(samplerFolder, sampler.getBasename()
                    + "-" + congressNum + ".html");
            sampler.outputHTML(htmlFile,
                    trainDebateIndices,
                    debateVoteData.getDocIds(),
                    debateVoteData.getRawSentences(),
                    debateVoteData.getAuthorTable(),
                    GovtrackUrl + congressNum + "/cr/");
            sampler.outputAuthorNodeMatrix(new File(samplerFolder, "author-node-matrix.txt"),
                    debateVoteData.getAuthorTable(), congressNum);
            sampler.analyzeAuthors(new File(samplerFolder, "analysis.author"),
                    debateVoteData.getAuthorTable());
            sampler.outputAuthorIssueDistribution(new File(samplerFolder, "author-issues-distrs.txt"),
                    debateVoteData.getAuthorTable(),
                    billData.getTopicVocab());

            File htmlFolder = new File(samplerFolder, sampler.getBasename()
                    + "-" + congressNum + "-authors");
            IOUtils.createFolder(htmlFolder);
            sampler.outputAuthorHTMLs(htmlFolder,
                    trainDebateIndices,
                    debateVoteData.getDocIds(),
                    debateVoteData.getRawSentences(),
                    debateVoteData.getAuthorTable(),
                    GovtrackUrl + congressNum + "/cr/");
        }
    }

    protected void runHybridSNLDAIdealPoint(File outputFolder) {
        // estimate seeded issues
        double[][] issuePhis;
        int K;
        if (cmd.hasOption("K")) {
            int V = debateVoteData.getWordVocab().size();
            K = Integer.parseInt(cmd.getOptionValue("K"));
            issuePhis = new double[K][V];
            for (int kk = 0; kk < K; kk++) {
                Arrays.fill(issuePhis[kk], 1.0 / V);
            }
        } else {
            issuePhis = estimateIssues();
            K = issuePhis.length;
        }
        int J = CLIUtils.getIntegerArgument(cmd, "J", 3);
        double[] alphas = CLIUtils.getDoubleArrayArgument(cmd, "alphas",
                new double[]{0.1, 0.1}, ",");
        double[] betas = CLIUtils.getDoubleArrayArgument(cmd, "betas",
                new double[]{10, 10, 0.1}, ",");
        double[] pis = CLIUtils.getDoubleArrayArgument(cmd, "pis",
                new double[]{0.2, 0.2}, ",");
        double[] gammas = CLIUtils.getDoubleArrayArgument(cmd, "gammas",
                new double[]{10, 1}, ",");
        double[] sigmas = CLIUtils.getDoubleArrayArgument(cmd, "sigmas",
                new double[]{2.5, 5}, ",");
        double sigma = CLIUtils.getDoubleArgument(cmd, "sigma", 1.0);
        double rho = CLIUtils.getDoubleArgument(cmd, "rho", 0.5);
        double lambda = CLIUtils.getDoubleArgument(cmd, "lambda", 2.5);
        boolean hasRootTopic = cmd.hasOption("roottopic");

        HybridSNLDAIdealPoint sampler = new HybridSNLDAIdealPoint();
        sampler.setVerbose(verbose);
        sampler.setDebug(debug);
        sampler.setLog(true);
        sampler.setReport(true);
        sampler.setWordVocab(debateVoteData.getWordVocab());
        sampler.setAuthorVocab(debateVoteData.getAuthorVocab());
        if (!cmd.hasOption("K")) {
            sampler.setLabelVocab(billData.getTopicVocab());
        }

        sampler.configure(outputFolder.getAbsolutePath(),
                debateVoteData.getWordVocab().size(), K, J,
                alphas, betas, pis, gammas,
                rho, sigma, sigmas, lambda, hasRootTopic,
                initState, paramOpt,
                burn_in, max_iters, sample_lag, report_interval);
        File samplerFolder = new File(outputFolder, sampler.getSamplerFolder());
        IOUtils.createFolder(samplerFolder);

        File teResultFolder = new File(samplerFolder, TEST_PREFIX + RESULT_FOLDER);
        File trResultFolder = new File(samplerFolder, TRAIN_PREFIX + RESULT_FOLDER);

        if (cmd.hasOption("train")) {
            sampler.setupData(trainDebateIndices,
                    debateVoteData.getWords(),
                    debateVoteData.getAuthors(),
                    votes, trainAuthorIndices, trainBillIndices,
                    trainVotes);
            sampler.initialize(issuePhis);
            sampler.metaIterate();
            sampler.outputTopicTopWords(new File(samplerFolder, TopWordFile), numTopWords);
            outputAuthorScore(new File(samplerFolder, AuthorScoreFile),
                    debateVoteData.getAuthorVocab(),
                    trainAuthorIndices,
                    trainVotes,
                    sampler.getUs(),
                    debateVoteData.getAuthorTable());
            outputVoteScores(new File(samplerFolder, VoteScoreFile),
                    debateVoteData.getVoteVocab(), this.trainBillIndices,
                    sampler.getXs(), sampler.getYs(),
                    debateVoteData.getVoteTable());

            SparseVector[] predictions = sampler.predictInMatrix();
            IOUtils.createFolder(trResultFolder);
            AbstractVotePredictor.outputPredictions(new File(trResultFolder, PREDICTION_FILE),
                    votes, predictions);
            AbstractModel.outputPerformances(new File(trResultFolder, RESULT_FILE),
                    AbstractVotePredictor.evaluateAll(votes, trainVotes, predictions));
            sampler.analyzeAuthors(new File(trResultFolder, "analysis.author"),
                    debateVoteData.getAuthorTable());
        }

        if (cmd.hasOption("testvote")) { // average over multiple test chains
            IOUtils.createFolder(teResultFolder);
            sampler.setupData(testDebateIndices,
                    debateVoteData.getWords(),
                    debateVoteData.getAuthors(),
                    null,
                    testAuthorIndices,
                    testBillIndices,
                    testVotes);
            SparseVector[] predictions = sampler.predictInMatrixMultiples();
            AbstractVotePredictor.outputPredictions(new File(teResultFolder, PREDICTION_FILE),
                    votes, predictions);
            AbstractModel.outputPerformances(new File(teResultFolder, RESULT_FILE),
                    AbstractVotePredictor.evaluateAll(votes, testVotes, predictions));
        }

        if (cmd.hasOption("testauthor")) {
            IOUtils.createFolder(teResultFolder);
            SparseVector[] predictions;
            if (cmd.hasOption("parallel")) {
                File iterPredFolderPath = new File(samplerFolder,
                        AbstractSampler.IterPredictionFolder);
                predictions = HybridSNLDAIdealPoint.parallelTest(testDebateIndices,
                        debateVoteData.getWords(),
                        debateVoteData.getAuthors(),
                        testAuthorIndices,
                        testVotes,
                        iterPredFolderPath,
                        sampler);

            } else {
                sampler.setupData(testDebateIndices,
                        debateVoteData.getWords(),
                        debateVoteData.getAuthors(),
                        null,
                        testAuthorIndices,
                        testBillIndices,
                        testVotes);
                predictions = sampler.test(null, null,
                        new File(samplerFolder, TEST_PREFIX + AssignmentFile));
                outputAuthorScore(new File(teResultFolder, AuthorScoreFile),
                        debateVoteData.getAuthorVocab(),
                        testAuthorIndices,
                        testVotes,
                        sampler.getPredictedUs(),
                        debateVoteData.getAuthorTable());

                sampler.outputTopicTopWords(new File(teResultFolder, TopWordFile), numTopWords);
                sampler.analyzeAuthors(new File(teResultFolder, "analysis.author"),
                        debateVoteData.getAuthorTable());
            }
            AbstractVotePredictor.outputPredictions(new File(teResultFolder, PREDICTION_FILE),
                    votes, predictions);
            AbstractModel.outputPerformances(new File(teResultFolder, RESULT_FILE),
                    AbstractVotePredictor.evaluateAll(votes, testVotes, predictions));
        }
    }

    protected void runHybridSNHDPIdealPoint(File outputFolder) {
        // estimate seeded issues
        double[][] issuePhis;
        int K;
        if (cmd.hasOption("K")) {
            int V = debateVoteData.getWordVocab().size();
            K = Integer.parseInt(cmd.getOptionValue("K"));
            issuePhis = new double[K][V];
            for (int kk = 0; kk < K; kk++) {
                Arrays.fill(issuePhis[kk], 1.0 / V);
            }
        } else {
            issuePhis = estimateIssues();
            K = issuePhis.length;
        }
        double[] localAlphas = CLIUtils.getDoubleArrayArgument(cmd, "local-alphas",
                new double[]{0.1, 0.1}, ",");
        double[] globalAlphas = CLIUtils.getDoubleArrayArgument(cmd, "global-alphas",
                new double[]{0.1, 0.1}, ",");
        double[] betas = CLIUtils.getDoubleArrayArgument(cmd, "betas",
                new double[]{10, 5, 0.1}, ",");
        double pi = CLIUtils.getDoubleArgument(cmd, "pi", 0.2);
        double[] gammas = CLIUtils.getDoubleArrayArgument(cmd, "gammas",
                new double[]{10, 1}, ",");
        double lambda = CLIUtils.getDoubleArgument(cmd, "lambda", 2.5);
        double l1 = CLIUtils.getDoubleArgument(cmd, "l1", 0.1);
        double l2 = CLIUtils.getDoubleArgument(cmd, "l2", 0.001);
        double[] sigmas = CLIUtils.getDoubleArrayArgument(cmd, "sigmas",
                new double[]{0.0, 1.0, 2.5}, ",");
        double sigma = CLIUtils.getDoubleArgument(cmd, "sigma", 2.5);
        double rho = CLIUtils.getDoubleArgument(cmd, "rho", 1.0);
        boolean hasRootTopic = cmd.hasOption("roottopic");

        HybridSNHDPIdealPoint sampler = new HybridSNHDPIdealPoint();
        sampler.setVerbose(verbose);
        sampler.setDebug(debug);
        sampler.setLog(true);
        sampler.setReport(true);
        sampler.setWordVocab(debateVoteData.getWordVocab());
        sampler.setAuthorVocab(debateVoteData.getAuthorVocab());
        if (!cmd.hasOption("K")) {
            sampler.setLabelVocab(billData.getTopicVocab());
        }

        PathAssumption pathAssumption = PathAssumption.MAXIMAL;
        String path = CLIUtils.getStringArgument(cmd, "path", "max");
        switch (path) {
            case "max":
                pathAssumption = PathAssumption.MAXIMAL;
                break;
            case "min":
                pathAssumption = PathAssumption.MINIMAL;
                break;
            default:
                throw new RuntimeException("Path assumption " + path + " not supported");
        }
        if (lambda > 0) {
            sampler.configure(outputFolder.getAbsolutePath(),
                    debateVoteData.getWordVocab().size(), K,
                    globalAlphas, localAlphas, betas, pi, gammas,
                    rho, sigmas, sigma, lambda, hasRootTopic,
                    initState, pathAssumption, paramOpt,
                    burn_in, max_iters, sample_lag, report_interval);
        } else {
            sampler.configure(outputFolder.getAbsolutePath(),
                    debateVoteData.getWordVocab().size(), K,
                    globalAlphas, localAlphas, betas, pi, gammas,
                    rho, sigmas, sigma, l1, l2, hasRootTopic,
                    initState, pathAssumption, paramOpt,
                    burn_in, max_iters, sample_lag, report_interval);
        }
        File samplerFolder = new File(outputFolder, sampler.getSamplerFolder());
        IOUtils.createFolder(samplerFolder);

        File teResultFolder = new File(samplerFolder, TEST_PREFIX + RESULT_FOLDER);
        File trResultFolder = new File(samplerFolder, TRAIN_PREFIX + RESULT_FOLDER);
        if (cmd.hasOption("train")) {
            sampler.setupData(trainDebateIndices,
                    debateVoteData.getWords(),
                    debateVoteData.getAuthors(),
                    votes, trainAuthorIndices, trainBillIndices,
                    trainVotes);
            sampler.initialize(issuePhis);
            sampler.metaIterate();
            sampler.outputTopicTopWords(new File(samplerFolder, TopWordFile), numTopWords);
            outputAuthorScore(new File(samplerFolder, AuthorScoreFile),
                    debateVoteData.getAuthorVocab(),
                    trainAuthorIndices,
                    trainVotes,
                    sampler.getUs(),
                    debateVoteData.getAuthorTable());
            outputVoteScores(new File(samplerFolder, VoteScoreFile),
                    debateVoteData.getVoteVocab(), this.trainBillIndices,
                    sampler.getXs(), sampler.getYs(),
                    debateVoteData.getVoteTable());

            SparseVector[] predictions = sampler.predictInMatrix();
            IOUtils.createFolder(trResultFolder);
            AbstractVotePredictor.outputPredictions(new File(trResultFolder, PREDICTION_FILE),
                    votes, predictions);
            AbstractModel.outputPerformances(new File(trResultFolder, RESULT_FILE),
                    AbstractVotePredictor.evaluateAll(votes, trainVotes, predictions));
            sampler.analyzeAuthors(new File(trResultFolder, "analysis.author"),
                    debateVoteData.getAuthorTable());
        }

        if (cmd.hasOption("testauthor")) {
            IOUtils.createFolder(teResultFolder);
            SparseVector[] predictions;
            if (cmd.hasOption("parallel")) {
                File iterPredFolderPath = new File(samplerFolder,
                        AbstractSampler.IterPredictionFolder);
                predictions = HybridSNHDPIdealPoint.parallelTest(testDebateIndices,
                        debateVoteData.getWords(),
                        debateVoteData.getAuthors(),
                        testAuthorIndices,
                        testVotes,
                        iterPredFolderPath,
                        sampler);

            } else {
                sampler.setupData(testDebateIndices,
                        debateVoteData.getWords(),
                        debateVoteData.getAuthors(),
                        null,
                        testAuthorIndices,
                        testBillIndices,
                        testVotes);
                predictions = sampler.test(null, null,
                        new File(samplerFolder, TEST_PREFIX + AssignmentFile));
                outputAuthorScore(new File(teResultFolder, AuthorScoreFile),
                        debateVoteData.getAuthorVocab(),
                        testAuthorIndices,
                        testVotes,
                        sampler.getPredictedUs(),
                        debateVoteData.getAuthorTable());

                sampler.outputTopicTopWords(new File(teResultFolder, TopWordFile), numTopWords);
                sampler.analyzeAuthors(new File(teResultFolder, "analysis.author"),
                        debateVoteData.getAuthorTable());
            }
            AbstractVotePredictor.outputPredictions(new File(teResultFolder, PREDICTION_FILE),
                    votes, predictions);
            AbstractModel.outputPerformances(new File(teResultFolder, RESULT_FILE),
                    AbstractVotePredictor.evaluateAll(votes, testVotes, predictions));
        }
    }

    protected void runSNHDPIdealPoint(File outputFolder) {
        // estimate seeded issues
        double[][] issuePhis;
        if (cmd.hasOption("K")) {
            int V = debateVoteData.getWordVocab().size();
            int K = Integer.parseInt(cmd.getOptionValue("K"));
            issuePhis = new double[K][V];
            for (int kk = 0; kk < K; kk++) {
                Arrays.fill(issuePhis[kk], 1.0 / V);
            }
        } else {
            issuePhis = estimateIssues();
        }
        double[] localAlphas = CLIUtils.getDoubleArrayArgument(cmd, "local-alphas",
                new double[]{0.1, 0.1}, ",");
        double[] globalAlphas = CLIUtils.getDoubleArrayArgument(cmd, "global-alphas",
                new double[]{0.1, 0.1}, ",");
        double[] betas = CLIUtils.getDoubleArrayArgument(cmd, "betas",
                new double[]{10, 5, 0.1}, ",");
        double[] gamma_means = CLIUtils.getDoubleArrayArgument(cmd, "gamma-means",
                new double[]{0.2, 0.2}, ",");
        double[] gamma_scales = CLIUtils.getDoubleArrayArgument(cmd, "gamma-scales",
                new double[]{10, 1}, ",");
        double mu = CLIUtils.getDoubleArgument(cmd, "mu", 0.0);
        double[] sigmas = CLIUtils.getDoubleArrayArgument(cmd, "sigmas",
                new double[]{0.0, 1.0, 2.5}, ",");
        double sigma = CLIUtils.getDoubleArgument(cmd, "sigma", 2.5);
        double rho = CLIUtils.getDoubleArgument(cmd, "rho", 1.0);
        boolean hasRootTopic = cmd.hasOption("roottopic");

        SNHDPIdealPoint sampler = new SNHDPIdealPoint();
        sampler.setVerbose(verbose);
        sampler.setDebug(debug);
        sampler.setLog(true);
        sampler.setReport(true);
        sampler.setWordVocab(debateVoteData.getWordVocab());
        sampler.setAuthorVocab(debateVoteData.getAuthorVocab());
        if (!cmd.hasOption("K")) {
            sampler.setLabelVocab(billData.getTopicVocab());
        }

        PathAssumption pathAssumption = PathAssumption.MAXIMAL;
        String path = CLIUtils.getStringArgument(cmd, "path", "max");
        switch (path) {
            case "max":
                pathAssumption = PathAssumption.MAXIMAL;
                break;
            case "min":
                pathAssumption = PathAssumption.MINIMAL;
                break;
            default:
                throw new RuntimeException("Path assumption " + path + " not supported");
        }

        sampler.configure(outputFolder.getAbsolutePath(),
                debateVoteData.getWordVocab().size(),
                issuePhis,
                globalAlphas, localAlphas, betas, gamma_means, gamma_scales,
                rho, mu, sigmas, sigma, hasRootTopic,
                initState, pathAssumption, paramOpt,
                burn_in, max_iters, sample_lag, report_interval);
        File samplerFolder = new File(outputFolder, sampler.getSamplerFolder());
        IOUtils.createFolder(samplerFolder);
        File teResultFolder = new File(samplerFolder, TEST_PREFIX + RESULT_FOLDER);
        File trResultFolder = new File(samplerFolder, TRAIN_PREFIX + RESULT_FOLDER);

        if (cmd.hasOption("train")) {
            sampler.setupData(trainDebateIndices,
                    debateVoteData.getWords(),
                    debateVoteData.getAuthors(),
                    votes, trainAuthorIndices, trainBillIndices,
                    trainVotes);
            sampler.initialize();
            sampler.iterate();
            sampler.outputTopicTopWords(new File(samplerFolder, TopWordFile), numTopWords);
            outputAuthorScore(new File(samplerFolder, AuthorScoreFile),
                    debateVoteData.getAuthorVocab(),
                    trainAuthorIndices,
                    trainVotes,
                    sampler.getUs(),
                    debateVoteData.getAuthorTable());
            outputVoteScores(new File(samplerFolder, VoteScoreFile),
                    debateVoteData.getVoteVocab(), this.trainBillIndices,
                    sampler.getXs(), sampler.getYs(),
                    debateVoteData.getVoteTable());
        }

        if (cmd.hasOption("testvote")) {
            IOUtils.createFolder(teResultFolder);
            sampler.setupData(testDebateIndices,
                    debateVoteData.getWords(),
                    debateVoteData.getAuthors(),
                    null,
                    testAuthorIndices,
                    testBillIndices,
                    testVotes);
            SparseVector[] predictions = null;
            int count = 0;
            File reportFolder = new File(sampler.getReportFolderPath());
            String[] files = reportFolder.list();
            for (String file : files) {
                if (!file.endsWith(".zip")) {
                    continue;
                }
                sampler.inputState(new File(reportFolder, file));
                SparseVector[] partPreds = sampler.predictInMatrix();
                if (predictions == null) {
                    predictions = partPreds;
                } else {
                    for (int aa = 0; aa < predictions.length; aa++) {
                        predictions[aa].add(partPreds[aa]);
                    }
                }
                count++;
            }
            for (SparseVector prediction : predictions) {
                prediction.scale(1.0 / count);
            }
            AbstractVotePredictor.outputPredictions(new File(teResultFolder, PREDICTION_FILE),
                    votes, predictions);
            AbstractModel.outputPerformances(new File(teResultFolder, RESULT_FILE),
                    AbstractVotePredictor.evaluateAll(votes, testVotes, predictions));
        }

        if (cmd.hasOption("testauthor")) {
            IOUtils.createFolder(teResultFolder);
            SparseVector[] predictions;
            if (cmd.hasOption("parallel")) {
                File iterPredFolderPath = new File(samplerFolder,
                        AbstractSampler.IterPredictionFolder);
                predictions = SNHDPIdealPoint.parallelTest(testDebateIndices,
                        debateVoteData.getWords(),
                        debateVoteData.getAuthors(),
                        testAuthorIndices,
                        testVotes,
                        iterPredFolderPath,
                        sampler);
            } else {
                sampler.setupData(testDebateIndices,
                        debateVoteData.getWords(),
                        debateVoteData.getAuthors(),
                        null,
                        testAuthorIndices,
                        testBillIndices,
                        testVotes);
                predictions = sampler.test(null, null,
                        new File(samplerFolder, TEST_PREFIX + AssignmentFile));
                outputAuthorScore(new File(teResultFolder, AuthorScoreFile),
                        debateVoteData.getAuthorVocab(),
                        testAuthorIndices,
                        testVotes,
                        sampler.getPredictedUs(),
                        debateVoteData.getAuthorTable());
            }
            AbstractVotePredictor.outputPredictions(new File(teResultFolder, PREDICTION_FILE),
                    votes, predictions);
            AbstractModel.outputPerformances(new File(teResultFolder, RESULT_FILE),
                    AbstractVotePredictor.evaluateAll(votes, testVotes, predictions));
        }
    }

    protected void runSNLDAMultIdealPoint(File outputFolder) {
//        double[][] issuePhis = estimateIssues();
//        int J = CLIUtils.getIntegerArgument(cmd, "J", 3);
//        double[] alphas = CLIUtils.getDoubleArrayArgument(cmd, "alphas",
//                new double[]{0.1, 0.1}, ",");
//        double[] betas = CLIUtils.getDoubleArrayArgument(cmd, "betas",
//                new double[]{10, 10, 0.1}, ",");
//        double[] gamma_means = CLIUtils.getDoubleArrayArgument(cmd, "gamma-means",
//                new double[]{0.2, 0.2}, ",");
//        double[] gamma_scales = CLIUtils.getDoubleArrayArgument(cmd, "gamma-scales",
//                new double[]{10, 1}, ",");
//        double mu = CLIUtils.getDoubleArgument(cmd, "mu", 0.0);
//        double[] sigmas = CLIUtils.getDoubleArrayArgument(cmd, "sigmas",
//                new double[]{0.0, 1.0, 2.5}, ",");
//        double l1 = CLIUtils.getDoubleArgument(cmd, "l1", 0.0);
//        double l2 = CLIUtils.getDoubleArgument(cmd, "l2", 1.0);
//        double lexl1 = CLIUtils.getDoubleArgument(cmd, "lexl1", 0.0);
//        double lexl2 = CLIUtils.getDoubleArgument(cmd, "lexl2", 0.0);
//        boolean hasRootTopic = cmd.hasOption("roottopic");
//
//        SNLDAMultIdealPoint sampler = new SNLDAMultIdealPoint();
//        sampler.setVerbose(verbose);
//        sampler.setDebug(debug);
//        sampler.setLog(true);
//        sampler.setReport(true);
//        sampler.setWordVocab(debateVoteData.getWordVocab());
//        sampler.setAuthorVocab(debateVoteData.getAuthorVocab());
//        sampler.setLabelVocab(billData.getTopicVocab());
//
//        PathAssumption pathAssumption = PathAssumption.MAXIMAL;
//        String path = CLIUtils.getStringArgument(cmd, "path", "max");
//        switch (path) {
//            case "max":
//                pathAssumption = PathAssumption.MAXIMAL;
//                break;
//            case "min":
//                pathAssumption = PathAssumption.MINIMAL;
//                break;
//            default:
//                throw new RuntimeException("Path assumption " + path + " not supported");
//        }
//
//        sampler.configure(outputFolder.getAbsolutePath(),
//                debateVoteData.getWordVocab().size(), J,
//                issuePhis, alphas, betas, gamma_means, gamma_scales,
//                mu, sigmas, l1, l2, lexl1, lexl2, hasRootTopic,
//                initState, pathAssumption, paramOpt,
//                burn_in, max_iters, sample_lag, report_interval);
//        File samplerFolder = new File(outputFolder, sampler.getSamplerFolder());
//        IOUtils.createFolder(samplerFolder);
//
//        if (cmd.hasOption("train")) {
//            sampler.setupData(trainDebateIndices,
//                    debateVoteData.getWords(),
//                    debateVoteData.getAuthors(),
//                    votes, trainAuthorIndices, trainBillIndices,
//                    trainVotes);
//            sampler.initialize();
//            sampler.iterate();
//            sampler.outputTopicTopWords(new File(samplerFolder, TopWordFile), numTopWords);
//        }
//
//        if (cmd.hasOption("testauthor")) {
//            File teResultFolder = new File(samplerFolder, TEST_PREFIX + RESULT_FOLDER);
//            IOUtils.createFolder(teResultFolder);
//
//            SparseVector[] predictions = sampler.test(null,
//                    testDebateIndices,
//                    debateVoteData.getWords(),
//                    debateVoteData.getAuthors(),
//                    testAuthorIndices,
//                    testVotes,
//                    null, null, null,
//                    new File(samplerFolder, TEST_PREFIX + "assignments.zip"));
//
//            String predFile = PREDICTION_FILE;
//            String resultFile = RESULT_FILE;
//            AbstractVotePredictor.outputPredictions(new File(teResultFolder, predFile),
//                    votes, predictions);
//            AbstractModel.outputPerformances(new File(teResultFolder, resultFile),
//                    AbstractVotePredictor.evaluate(votes, testVotes, predictions));
//        }
    }

    protected void combineModel(File outputFolder) {
        ArrayList<String> modelNames = new ArrayList<>();
        modelNames.add("logreg-OWLQN-TFIDF_l1-0_l2-2.5");
        modelNames.add("RANDOM_SNLDA-ideal-point_B-1000_M-2000_L-100_K-25_J-4_"
                + "a-0.1-0.1_b-1-0.5-0.1_gm-0.2-0.2_gs-100-10_r-1_m-0_s-2.5_opt-false_rt-false_MAXIMAL");

        ArrayList<Double> modelWeights = new ArrayList<>();
        modelWeights.add(1.0);
        modelWeights.add(1.0);

        String combinedModel = "logreg-TFIDF-SNLDA-K25-J4";
        combinePredictions(outputFolder, modelNames, modelWeights, combinedModel);
    }

    protected void combineModelAuthorDependent(File outputFolder) {
        double alpha = 0.005;
        double mu = 1000;
        ArrayList<String> modelNames = new ArrayList<>();
        modelNames.add("logreg-OWLQN-TFIDF_l1-0_l2-2.5");
        modelNames.add("RANDOM_SNLDA-ideal-point_B-1000_M-2000_L-100_K-25_J-4_"
                + "a-0.1-0.1_b-1-0.5-0.1_gm-0.2-0.2_gs-100-10_r-1_m-0_s-2.5_opt-false_rt-false_MAXIMAL");

        SparseCount authorTokenCounts = new SparseCount();
        for (int dd : testDebateIndices) {
            int author = debateVoteData.getAuthors()[dd];
            authorTokenCounts.changeCount(author, debateVoteData.getWords()[dd].length);
        }
        HashMap<Integer, ArrayList<Double>> modelWeightPerAuthors = new HashMap<>();
        for (int author : authorTokenCounts.getIndices()) {
            int tokenCount = authorTokenCounts.getCount(author);
            double val = sigmoid(tokenCount, alpha, mu);
            ArrayList<Double> modelWeights = new ArrayList<>();
            modelWeights.add(val); // logistic regression
            modelWeights.add(1.0 - val);
            modelWeightPerAuthors.put(author, modelWeights);
        }

        String combinedModel = "logreg-TFIDF-SNLDA-K25-J4-author-specific";
        combinePredictions(outputFolder, modelNames, modelWeightPerAuthors, combinedModel);
    }

    private double sigmoid(int count, double alpha, double mu) {
        return 1.0 / (1.0 + Math.exp(-alpha * (count - mu)));
    }

    protected void combinePredictions(File outputFolder,
            ArrayList<String> modelNames,
            HashMap<Integer, ArrayList<Double>> modelWeightsPerAuthor,
            String combinedModelName) {
        ArrayList<SparseVector[]> predictionList = new ArrayList<>();
        for (String modelName : modelNames) {
            File teResultFolder = new File(new File(outputFolder, modelName), TEST_PREFIX + RESULT_FOLDER);
            File tePredictionFile = new File(teResultFolder, PREDICTION_FILE);
            SparseVector[] predictions = AbstractVotePredictor.inputPredictions(tePredictionFile);
            if (verbose) {
                logln("--- Loading prediction from " + tePredictionFile);
            }
            predictionList.add(predictions);
        }

        if (testBillIndices == null) {
            testBillIndices = new ArrayList<>();
            for (int bb = 0; bb < testVotes[0].length; bb++) {
                testBillIndices.add(bb);
            }
        }
        int A = testAuthorIndices.size();
        int B = testBillIndices.size();

        SparseVector[] predictions = new SparseVector[testVotes.length];
        for (int aa = 0; aa < A; aa++) {
            int author = testAuthorIndices.get(aa);
            ArrayList<Double> modelWeights = modelWeightsPerAuthor.get(author);
            double totalWeight = StatUtils.sum(modelWeights);
            predictions[author] = new SparseVector(testVotes[author].length);
            for (int bb = 0; bb < B; bb++) {
                int bill = testBillIndices.get(bb);
                if (testVotes[author][bill]) {
                    double val = 0.0;
                    for (int ii = 0; ii < predictionList.size(); ii++) {
                        val += predictionList.get(ii)[author].get(bill) * modelWeights.get(ii);
                    }
                    predictions[author].set(bill, val / totalWeight);
                }
            }
        }

        File teResultFolder = new File(new File(outputFolder, combinedModelName),
                TEST_PREFIX + RESULT_FOLDER);
        IOUtils.createFolder(teResultFolder);
        AbstractVotePredictor.outputPredictions(new File(teResultFolder, PREDICTION_FILE),
                votes, predictions);
        AbstractModel.outputPerformances(new File(teResultFolder, RESULT_FILE),
                AbstractVotePredictor.evaluateAll(votes, testVotes, predictions));
    }

    protected void combinePredictions(File outputFolder,
            ArrayList<String> modelNames,
            ArrayList<Double> modelWeights,
            String combinedModelName) {
        ArrayList<SparseVector[]> predictionList = new ArrayList<>();
        for (String modelName : modelNames) {
            File teResultFolder = new File(new File(outputFolder, modelName), TEST_PREFIX + RESULT_FOLDER);
            File tePredictionFile = new File(teResultFolder, PREDICTION_FILE);
            SparseVector[] predictions = AbstractVotePredictor.inputPredictions(tePredictionFile);
            if (verbose) {
                logln("--- Loading prediction from " + tePredictionFile);
            }
            predictionList.add(predictions);
        }

        if (testBillIndices == null) {
            testBillIndices = new ArrayList<>();
            for (int bb = 0; bb < testVotes[0].length; bb++) {
                testBillIndices.add(bb);
            }
        }
        int A = testAuthorIndices.size();
        int B = testBillIndices.size();

        double totalWeight = StatUtils.sum(modelWeights);
        SparseVector[] predictions = new SparseVector[testVotes.length];
        for (int aa = 0; aa < A; aa++) {
            int author = testAuthorIndices.get(aa);
            predictions[author] = new SparseVector(testVotes[author].length);
            for (int bb = 0; bb < B; bb++) {
                int bill = testBillIndices.get(bb);
                if (testVotes[author][bill]) {
                    double val = 0.0;
                    for (int ii = 0; ii < predictionList.size(); ii++) {
                        val += predictionList.get(ii)[author].get(bill) * modelWeights.get(ii);
                    }
                    predictions[author].set(bill, val / totalWeight);
                }
            }
        }

        File teResultFolder = new File(new File(outputFolder, combinedModelName),
                TEST_PREFIX + RESULT_FOLDER);
        IOUtils.createFolder(teResultFolder);
        AbstractVotePredictor.outputPredictions(new File(teResultFolder, PREDICTION_FILE),
                votes, predictions);
        AbstractModel.outputPerformances(new File(teResultFolder, RESULT_FILE),
                AbstractVotePredictor.evaluateAll(votes, testVotes, predictions));
    }

    /**
     * Estimate a topic (i.e., a distribution over words) for each issue in the
     * Policy Agenda codebook using coded data from Congressional Bill Project.
     * This is mainly used for obtain prior for first-level nodes in the
     * hierarchy.
     *
     * @return Prior
     */
    protected double[][] estimateIssues() {
        int L = billData.getTopicVocab().size();
        int V = billData.getWordVocab().size();
        TFEstimator estimator = new TFEstimator(billData.getWords(),
                billData.getTopics(), L, V);
        double[][] priors = estimator.getPriors();

        // remove "Other" issue
        double[][] newPriors = new double[L - 1][V];
        int count = 0;
        for (int ll = 0; ll < L; ll++) {
            if (billData.getTopicVocab().get(ll).equals("Other")) {
                continue;
            }
            newPriors[count++] = priors[ll];
        }

        if (verbose) {
            displayTopics(policyAgendaIssues, debateVoteData.getWordVocab(), newPriors);
        }

        // to avoid 0.0 probabilities
        for (double[] prior : priors) {
            for (int vv = 0; vv < prior.length; vv++) {
                prior[vv] = (prior[vv] + 1.0 / V) / 2;
            }
        }

        return newPriors;
    }

    /**
     * Display prior topics.
     *
     * @param labelVocab
     * @param wordVocab
     * @param phis
     */
    private static void displayTopics(
            ArrayList<String> labelVocab,
            ArrayList<String> wordVocab,
            double[][] phis) {
        for (int ll = 0; ll < phis.length; ll++) {
            ArrayList<RankingItem<Integer>> rankWords = new ArrayList<>();
            for (int vv = 0; vv < phis[ll].length; vv++) {
                rankWords.add(new RankingItem<Integer>(vv, phis[ll][vv]));
            }
            Collections.sort(rankWords);

            if (labelVocab != null && ll < labelVocab.size()) {
                System.out.println("Topic " + ll + ": " + labelVocab.get(ll));
            } else {
                System.out.println("Topic " + ll);
            }

            for (int ii = 0; ii < 20; ii++) {
                RankingItem<Integer> item = rankWords.get(ii);
                System.out.print("\t" + wordVocab.get(item.getObject())
                        + ":" + MiscUtils.formatDouble(item.getPrimaryValue()));
            }
            System.out.println("\n");

            for (int ii = 0; ii < 20; ii++) {
                RankingItem<Integer> item = rankWords.get(ii);
                System.out.print("; " + wordVocab.get(item.getObject()));
            }

            System.out.println("\n");
        }
    }

    /**
     * Output author scores.
     *
     * @param outputFile Output file
     * @param authorVocab Author vocabulary
     * @param authorIndices List of selected authors
     * @param voteMask
     * @param authorScores Learned author scores
     * @param authorTable Author information
     */
    protected void outputAuthorScore(File outputFile,
            ArrayList<String> authorVocab,
            ArrayList<Integer> authorIndices,
            boolean[][] voteMask,
            double[] authorScores,
            HashMap<String, Author> authorTable) {
        if (verbose) {
            logln("Outputing author scores to " + outputFile);
        }
        if (authorIndices == null) {
            throw new RuntimeException("null authorIndices");
        }
        if (authorScores == null) {
            throw new RuntimeException("null authorScores");
        }
        try {
            BufferedWriter writer = IOUtils.getBufferedWriter(outputFile);
            writer.write("Index\tID\tScore\tNumWithVotes\tNumAgainstVotes"
                    + "\tFreedomWorksID\tICPSRID\tName\tParty\tNominateScore"
                    + "\tTPCaucus\tFWEndorsement\tTPExpress\tSPEndorsement\tGroup\n");
            for (int ii = 0; ii < authorIndices.size(); ii++) {
                int aa = authorIndices.get(ii);
                int withCount = 0;
                int againstCount = 0;
                for (int bb = 0; bb < votes[aa].length; bb++) {
                    if (voteMask[aa][bb]) {
                        if (votes[aa][bb] == Vote.WITH) {
                            withCount++;
                        } else if (votes[aa][bb] == Vote.AGAINST) {
                            againstCount++;
                        }
                    }
                }
                String aid = authorVocab.get(aa);
                writer.write(aa + "\t" + aid
                        + "\t" + authorScores[ii]
                        + "\t" + withCount
                        + "\t" + againstCount);
                if (authorTable != null) {
                    Author author = authorTable.get(aid);
                    String icpsrid = author.getProperty(GTLegislator.ICPSRID);
                    writer.write("\t" + author.getProperty(GTLegislator.FW_ID)
                            + "\t" + icpsrid
                            + "\t" + author.getProperty(GTLegislator.NAME)
                            + "\t" + author.getProperty(GTLegislator.PARTY)
                            + "\t" + author.getProperty(GTLegislator.NOMINATE_SCORE1)
                            + "\t" + (teapartyCaucusMapping.get(icpsrid) == null ? "Non-member"
                                    : (teapartyCaucusMapping.get(icpsrid) == 0 ? "Non-member" : "Member"))
                            + "\t" + (fwEndorsementMapping.get(icpsrid) == null ? "0" : fwEndorsementMapping.get(icpsrid))
                            + "\t" + (tpExpressMapping.get(icpsrid) == null ? "0" : tpExpressMapping.get(icpsrid))
                            + "\t" + (spEndorsementMapping.get(icpsrid) == null ? "0" : spEndorsementMapping.get(icpsrid))
                    );

                    String party = author.getProperty(GTLegislator.PARTY);
                    if (party.startsWith("D")) {
                        writer.write("\tDem.");
                    } else if (party.startsWith("R")) {
                        Integer tpc = teapartyCaucusMapping.get(icpsrid);
                        if (tpc == null || tpc == 0) {
                            writer.write("\tRep. Non-member");
                        } else {
                            writer.write("\tRep. Member");
                        }
                    }
                }
                writer.write("\n");
            }
            writer.close();
        } catch (IOException e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while outputing to " + outputFile);
        }
    }

    /**
     * Output author scores.
     *
     * @param outputFile Output file
     * @param authorVocab Author vocabulary
     * @param authorIndices List of selected authors
     * @param voteMask
     * @param authorMultIdealPoint Learned author scores
     * @param authorTable Author information
     */
    protected void outputAuthorScore(File outputFile,
            ArrayList<String> authorVocab,
            ArrayList<Integer> authorIndices,
            boolean[][] voteMask,
            double[] authorSingleIdealPoint,
            double[][] authorMultIdealPoint,
            HashMap<String, Author> authorTable) {
        if (verbose) {
            logln("Outputing author scores to " + outputFile);
        }
        if (authorIndices == null) {
            throw new RuntimeException("null authorIndices");
        }
        if (authorMultIdealPoint == null) {
            throw new RuntimeException("null authorScores");
        }
        if (authorMultIdealPoint[0].length != policyAgendaIssues.size()) {
            throw new MismatchRuntimeException(policyAgendaIssues.size(), authorMultIdealPoint[0].length);
        }
        try {
            BufferedWriter writer = IOUtils.getBufferedWriter(outputFile);
            writer.write("Index\tID\tNumWithVotes\tNumAgainstVotes"
                    + "\tFreedomWorksID\tName\tParty\tNominateScore");
            for (String policyAgendaIssue : policyAgendaIssues) {
                writer.write("\t\"" + policyAgendaIssue + "\"");
            }
            if (authorSingleIdealPoint != null) {
                writer.write("\tSingleIdealPoint");
            }
            writer.write("\tTPCaucus\tFWEndorsement\tTPExpress\tSPEndorsement\tGroup");
            writer.write("\n");

            for (int ii = 0; ii < authorIndices.size(); ii++) {
                int aa = authorIndices.get(ii);
                int withCount = 0;
                int againstCount = 0;
                for (int bb = 0; bb < votes[aa].length; bb++) {
                    if (voteMask[aa][bb]) {
                        if (votes[aa][bb] == Vote.WITH) {
                            withCount++;
                        } else if (votes[aa][bb] == Vote.AGAINST) {
                            againstCount++;
                        }
                    }
                }
                String aid = authorVocab.get(aa);
                writer.write(aa + "\t" + aid
                        + "\t" + withCount
                        + "\t" + againstCount);

                Author author = authorTable.get(aid);
                String icpsrid = author.getProperty(GTLegislator.ICPSRID);
                writer.write("\t" + author.getProperty(GTLegislator.FW_ID)
                        + "\t" + author.getProperty(GTLegislator.NAME)
                        + "\t" + author.getProperty(GTLegislator.PARTY)
                        + "\t" + author.getProperty(GTLegislator.NOMINATE_SCORE1)
                );
                for (int ll = 0; ll < policyAgendaIssues.size(); ll++) {
                    writer.write("\t" + authorMultIdealPoint[ii][ll]);
                }

                if (authorSingleIdealPoint != null) {
                    writer.write("\t" + authorSingleIdealPoint[ii]);
                }
                writer.write("\t" + (teapartyCaucusMapping.get(icpsrid) == null ? "Non-membership"
                        : (teapartyCaucusMapping.get(icpsrid) == 0 ? "Non-membership" : "Membership"))
                        + "\t" + (fwEndorsementMapping.get(icpsrid) == null ? "0" : fwEndorsementMapping.get(icpsrid))
                        + "\t" + (tpExpressMapping.get(icpsrid) == null ? "0" : tpExpressMapping.get(icpsrid))
                        + "\t" + (spEndorsementMapping.get(icpsrid) == null ? "0" : spEndorsementMapping.get(icpsrid)));

                String party = author.getProperty(GTLegislator.PARTY);
                if (party.startsWith("D")) {
                    writer.write("\tDem.");
                } else if (party.startsWith("R")) {
                    Integer tpc = teapartyCaucusMapping.get(icpsrid);
                    if (tpc == null || tpc == 0) {
                        writer.write("\tRep. Non-member");
                    } else {
                        writer.write("\tRep. Member");
                    }
                }
                writer.write("\n");
            }
            writer.close();
        } catch (IOException e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while outputing to " + outputFile);
        }
    }

    /**
     * Output author scores.
     *
     * @param outputFile Output file
     * @param authorVocab Author vocabulary
     * @param authorIndices List of selected authors
     * @param voteMask
     * @param authorMultIdealPoint Learned author scores
     * @param authorTable Author information
     */
    protected void outputAuthorScore(File outputFile,
            ArrayList<String> authorVocab,
            ArrayList<Integer> authorIndices,
            boolean[][] voteMask,
            double[] authorSingleIdealPoint,
            double[][] authorMultIdealPoint,
            SparseCount[] authorTopicCounts,
            HashMap<String, Author> authorTable) {
        if (verbose) {
            logln("Outputing author scores to " + outputFile);
        }
        if (authorIndices == null) {
            throw new RuntimeException("null authorIndices");
        }
        if (authorMultIdealPoint == null) {
            throw new RuntimeException("null authorScores");
        }
        if (authorMultIdealPoint[0].length != policyAgendaIssues.size()) {
            throw new MismatchRuntimeException(policyAgendaIssues.size(), authorMultIdealPoint[0].length);
        }
        try {
            BufferedWriter writer = IOUtils.getBufferedWriter(outputFile);
            writer.write("Index\tID\tNumWithVotes\tNumAgainstVotes"
                    + "\tFreedomWorksID\tName\tParty\tNominateScore");
            for (String policyAgendaIssue : policyAgendaIssues) {
                writer.write("\t\"" + policyAgendaIssue + "\"");
            }
            if (authorSingleIdealPoint != null) {
                writer.write("\tSingleIdealPoint");
            }
            writer.write("\tTPCaucus\tFWEndorsement\tTPExpress\tSPEndorsement\tGroup");
            writer.write("\n");

            for (int ii = 0; ii < authorIndices.size(); ii++) {
                int aa = authorIndices.get(ii);
                int withCount = 0;
                int againstCount = 0;
                for (int bb = 0; bb < votes[aa].length; bb++) {
                    if (voteMask[aa][bb]) {
                        if (votes[aa][bb] == Vote.WITH) {
                            withCount++;
                        } else if (votes[aa][bb] == Vote.AGAINST) {
                            againstCount++;
                        }
                    }
                }
                String aid = authorVocab.get(aa);
                writer.write(aa + "\t" + aid
                        + "\t" + withCount
                        + "\t" + againstCount);

                Author author = authorTable.get(aid);
                String icpsrid = author.getProperty(GTLegislator.ICPSRID);
                writer.write("\t" + author.getProperty(GTLegislator.FW_ID)
                        + "\t" + author.getProperty(GTLegislator.NAME)
                        + "\t" + author.getProperty(GTLegislator.PARTY)
                        + "\t" + author.getProperty(GTLegislator.NOMINATE_SCORE1)
                );
                for (int ll = 0; ll < policyAgendaIssues.size(); ll++) {
                    writer.write("\t" + authorMultIdealPoint[ii][ll]
                            + " (" + authorTopicCounts[ii].getCount(ll) + ")");
                }

                if (authorSingleIdealPoint != null) {
                    writer.write("\t" + authorSingleIdealPoint[ii]);
                }
                writer.write("\t" + (teapartyCaucusMapping.get(icpsrid) == null ? "0" : teapartyCaucusMapping.get(icpsrid))
                        + "\t" + (fwEndorsementMapping.get(icpsrid) == null ? "0" : fwEndorsementMapping.get(icpsrid))
                        + "\t" + (tpExpressMapping.get(icpsrid) == null ? "0" : tpExpressMapping.get(icpsrid))
                        + "\t" + (spEndorsementMapping.get(icpsrid) == null ? "0" : spEndorsementMapping.get(icpsrid)));

                String party = author.getProperty(GTLegislator.PARTY);
                if (party.startsWith("D")) {
                    writer.write("\tDem.");
                } else if (party.startsWith("R  ")) {
                    Integer tpc = teapartyCaucusMapping.get(icpsrid);
                    if (tpc == null || tpc == 0) {
                        writer.write("\tRep. Non-member");
                    } else {
                        writer.write("\tRep. Member");
                    }
                }
                writer.write("\n");
            }
            writer.close();
        } catch (IOException e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while outputing to " + outputFile);
        }
    }

    protected void outputAuthorScore(File outputFile,
            ArrayList<String> authorVocab,
            ArrayList<Integer> authorIndices,
            boolean[][] voteMask,
            double[] authorScores,
            HashMap<String, Author> authorTable,
            double[][] authorTopicDists,
            double[] topicScores,
            ArrayList<String> topicNames) {
        if (verbose) {
            logln("Outputing author scores to " + outputFile);
        }
        if (authorIndices == null) {
            throw new RuntimeException("null authorIndices");
        }
        if (authorScores == null) {
            throw new RuntimeException("null authorScores");
        }

        // rank topics
        RankingItemList<Integer> rankTopics = new RankingItemList<>();
        for (int kk = 0; kk < topicNames.size(); kk++) {
            rankTopics.addRankingItem(kk, topicScores[kk]);
        }
        rankTopics.sortAscending();

        // rank authors
        RankingItemList<Integer> rankAuthors = new RankingItemList<>();
        for (int ii = 0; ii < authorIndices.size(); ii++) {
            rankAuthors.addRankingItem(ii, authorScores[ii]);
        }
        rankAuthors.sortAscending();

        try {
            BufferedWriter writer = IOUtils.getBufferedWriter(outputFile);
            writer.write("Name\tScore");
            for (int kk = 0; kk < rankTopics.size(); kk++) {
                RankingItem<Integer> rankTopic = rankTopics.getRankingItem(kk);
                writer.write("\t\"" + topicNames.get(rankTopic.getObject())
                        + " (" + MiscUtils.formatDouble(rankTopic.getPrimaryValue()) + ")\"");
            }
            writer.write("\n");

            for (int jj = 0; jj < authorIndices.size(); jj++) {
                RankingItem<Integer> rankAuthor = rankAuthors.getRankingItem(jj);
                int ii = rankAuthor.getObject();
                int aa = authorIndices.get(ii);
                int withCount = 0;
                int againstCount = 0;
                for (int bb = 0; bb < votes[aa].length; bb++) {
                    if (voteMask[aa][bb]) {
                        if (votes[aa][bb] == Vote.WITH) {
                            withCount++;
                        } else if (votes[aa][bb] == Vote.AGAINST) {
                            againstCount++;
                        }
                    }
                }
                String aid = authorVocab.get(aa);
                Author author = authorTable.get(aid);
                writer.write(author.getProperty(GTLegislator.NAME)
                        + " (" + MiscUtils.formatDouble(authorScores[ii])
                        + ", " + withCount + "/" + againstCount + ")");
                writer.write("\t" + authorScores[ii]);

                for (int kk = 0; kk < rankTopics.size(); kk++) {
                    int topicIdx = rankTopics.getRankingItem(kk).getObject();
                    writer.write("\t" + authorTopicDists[ii][topicIdx]);
                }

                writer.write("\n");
            }
            writer.close();
        } catch (IOException e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while outputing to " + outputFile);
        }
    }

    /**
     * Output vote scores.
     *
     * @param outputFile Output file
     * @param voteVocab List of votes
     * @param voteIndices
     * @param xs
     * @param ys
     * @param voteTable Vote information
     */
    public void outputVoteScores(File outputFile,
            ArrayList<String> voteVocab, ArrayList<Integer> voteIndices,
            double[] xs, double[] ys,
            HashMap<String, Vote> voteTable) {
        if (verbose) {
            logln("Outputing vote scores to " + outputFile);
        }
        this.trainBillIndices = voteIndices;
        if (voteIndices == null) {
            this.trainBillIndices = new ArrayList<>();
            for (int bb = 0; bb < voteVocab.size(); bb++) {
                this.trainBillIndices.add(bb);
            }
        }
        try {
            BufferedWriter writer = IOUtils.getBufferedWriter(outputFile);
            for (int jj = 0; jj < this.trainBillIndices.size(); jj++) {
                int bb = this.trainBillIndices.get(jj);
                int withCount = 0;
                int againstCount = 0;
                for (int aa = 0; aa < votes.length; aa++) {
                    if (trainVotes[aa][bb]) {
                        if (votes[aa][bb] == Vote.WITH) {
                            withCount++;
                        } else if (votes[aa][bb] == Vote.AGAINST) {
                            againstCount++;
                        }
                    }
                }

                String vid = voteVocab.get(bb);
                writer.write(bb + "\t" + vid
                        + "\t" + xs[jj]
                        + "\t" + ys[jj]
                        + "\t" + withCount
                        + "\t" + againstCount);
                if (voteTable != null) {
                    Vote vote = voteTable.get(vid);
                    writer.write("\t" + vote.getProperty(FWBill.FW_VOTE_PREFERRED)
                            + "\t" + vote.getProperty(FWBill.TITLE));
                }
                writer.write("\n");
            }
            writer.close();
        } catch (IOException e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while outputing to " + outputFile);
        }
    }

    /**
     * Output vote scores.
     *
     * @param outputFile Output file
     * @param voteVocab List of votes
     * @param voteIndices
     * @param xs
     * @param ys
     * @param voteTable Vote information
     */
    public void outputVoteScores(File outputFile,
            ArrayList<String> voteVocab, ArrayList<Integer> voteIndices,
            double[][] multXs,
            double[] xs, double[] ys,
            HashMap<String, Vote> voteTable) {
        if (verbose) {
            logln("Outputing vote scores to " + outputFile);
        }
        this.trainBillIndices = voteIndices;
        if (voteIndices == null) {
            this.trainBillIndices = new ArrayList<>();
            for (int bb = 0; bb < voteVocab.size(); bb++) {
                this.trainBillIndices.add(bb);
            }
        }
        try {
            BufferedWriter writer = IOUtils.getBufferedWriter(outputFile);
            writer.write("Index\tId\tWith\tTPWith\tNTPWith\tAgainst\tTPAgainst\t"
                    + "NTPAgainst\tPreferredOutcome\tTitle");
            for (String issue : policyAgendaIssues) {
                writer.write("\t" + issue);
            }
            if (xs != null) {
                writer.write("\tSingleX\tSingleY");
            }
            writer.write("\n");

            for (int jj = 0; jj < this.trainBillIndices.size(); jj++) {
                int bb = this.trainBillIndices.get(jj);
                int withCount = 0;
                int againstCount = 0;
                int tpWith = 0;
                int ntpWith = 0;
                int tpAgainst = 0;
                int ntpAgainst = 0;
                for (int aa = 0; aa < votes.length; aa++) {
                    String authorId = debateVoteData.getAuthorVocab().get(aa);
                    Author author = debateVoteData.getAuthorTable().get(authorId);
                    String icpsrId = author.getProperty(GTLegislator.ICPSRID);
                    int tpc = getTeaPartyCaucus(icpsrId);

                    if (trainVotes[aa][bb]) {
                        if (votes[aa][bb] == Vote.WITH) {
                            withCount++;
                            if (tpc == 1) {
                                tpWith++;
                            } else {
                                ntpWith++;
                            }

                        } else if (votes[aa][bb] == Vote.AGAINST) {
                            againstCount++;
                            if (tpc == 1) {
                                tpAgainst++;
                            } else {
                                ntpAgainst++;
                            }
                        }
                    }
                }

                String vid = voteVocab.get(bb);
                writer.write(bb + "\t" + vid
                        + "\t" + withCount
                        + "\t" + tpWith
                        + "\t" + ntpWith
                        + "\t" + againstCount
                        + "\t" + tpAgainst
                        + "\t" + ntpAgainst
                );
                if (voteTable != null) {
                    Vote vote = voteTable.get(vid);
                    writer.write("\t" + vote.getProperty(FWBill.FW_VOTE_PREFERRED)
                            + "\t" + vote.getProperty(FWBill.TITLE));
                }

                for (int kk = 0; kk < policyAgendaIssues.size(); kk++) {
                    writer.write("\t" + multXs[jj][kk]);
                }
                if (xs != null) {
                    writer.write("\t" + xs[jj] + "\t" + ys[jj]);
                }
                writer.write("\n");
            }
            writer.close();
        } catch (IOException e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while outputing to " + outputFile);
        }
    }

    public void outputVoteScores(File outputFile,
            ArrayList<String> voteVocab, ArrayList<Integer> voteIndices,
            double[][] thetas,
            HashMap<String, Vote> voteTable, int topK) {
        if (verbose) {
            logln("Outputing vote scores to " + outputFile);
        }
        this.trainBillIndices = voteIndices;
        if (voteIndices == null) {
            this.trainBillIndices = new ArrayList<>();
            for (int bb = 0; bb < voteVocab.size(); bb++) {
                this.trainBillIndices.add(bb);
            }
        }
        try {
            BufferedWriter writer = IOUtils.getBufferedWriter(outputFile);
            writer.write("Index\tId\tWith\tTPWith\tNTPWith\tAgainst\tTPAgainst\t"
                    + "NTPAgainst\tPreferredOutcome\tTitle");
            for (int ii = 0; ii < topK; ii++) {
                writer.write("\tTop-" + ii + "\tProb-" + ii);
            }
            writer.write("\n");

            for (int jj = 0; jj < this.trainBillIndices.size(); jj++) {
                int bb = this.trainBillIndices.get(jj);
                int withCount = 0;
                int againstCount = 0;
                int tpWith = 0;
                int ntpWith = 0;
                int tpAgainst = 0;
                int ntpAgainst = 0;
                for (int aa = 0; aa < votes.length; aa++) {
                    String authorId = debateVoteData.getAuthorVocab().get(aa);
                    Author author = debateVoteData.getAuthorTable().get(authorId);
                    String icpsrId = author.getProperty(GTLegislator.ICPSRID);
                    int tpc = getTeaPartyCaucus(icpsrId);

                    if (trainVotes[aa][bb]) {
                        if (votes[aa][bb] == Vote.WITH) {
                            withCount++;
                            if (tpc == 1) {
                                tpWith++;
                            } else {
                                ntpWith++;
                            }

                        } else if (votes[aa][bb] == Vote.AGAINST) {
                            againstCount++;
                            if (tpc == 1) {
                                tpAgainst++;
                            } else {
                                ntpAgainst++;
                            }
                        }
                    }
                }

                String vid = voteVocab.get(bb);
                writer.write(bb + "\t" + vid
                        + "\t" + withCount
                        + "\t" + tpWith
                        + "\t" + ntpWith
                        + "\t" + againstCount
                        + "\t" + tpAgainst
                        + "\t" + ntpAgainst
                );
                if (voteTable != null) {
                    Vote vote = voteTable.get(vid);
                    writer.write("\t" + vote.getProperty(FWBill.FW_VOTE_PREFERRED)
                            + "\t" + vote.getProperty(FWBill.TITLE));
                }
                ArrayList<RankingItem<String>> rankLabels = new ArrayList<>();
                for (int kk = 0; kk < policyAgendaIssues.size(); kk++) {
                    rankLabels.add(new RankingItem<String>(policyAgendaIssues.get(kk), thetas[jj][kk]));
                }
                Collections.sort(rankLabels);
                for (int ii = 0; ii < topK; ii++) {
                    RankingItem<String> rankLabel = rankLabels.get(ii);
                    writer.write("\t" + rankLabel.getObject()
                            + "\t" + rankLabel.getPrimaryValue());
                }
                writer.write("\n");
            }
            writer.close();
        } catch (IOException e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while outputing to " + outputFile);
        }
    }

    @Override
    public void evaluate() {
        if (verbose) {
            logln("Evaluating ...");
        }
        File resultFolder = new File(new File(experimentPath, congressNum),
                getConfiguredExptFolder());
        try {
            evaluate(resultFolder.getAbsolutePath(),
                    null, numFolds,
                    TEST_PREFIX,
                    RESULT_FILE);
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while evaluating");
        }
    }

    protected void analyzeErrorMultipleModels(File foldFolder) {
        if (verbose) {
            logln("Analyzing error from multiple models in " + formatFolder);
        }
        ArrayList<String> modelNames = new ArrayList<>();
        modelNames.add("logreg-OWLQN-TFIDF_l1-0_l2-2.5");
        modelNames.add("RANDOM_SNLDA-ideal-point_B-1000_M-2000_L-100_K-25_J-4_"
                + "a-0.1-0.1_b-1-0.5-0.1_gm-0.2-0.2_gs-100-10_r-1_m-0_s-2.5_opt-false_rt-false_MAXIMAL");

        ArrayList<SparseVector[]> predictionList = new ArrayList<>();
        for (String modelName : modelNames) {
            File teResultFolder = new File(new File(foldFolder, modelName), TEST_PREFIX + RESULT_FOLDER);
            File tePredictionFile = new File(teResultFolder, PREDICTION_FILE);
            SparseVector[] predictions = AbstractVotePredictor.inputPredictions(tePredictionFile);
            if (verbose) {
                logln("--- Loading prediction from " + tePredictionFile);
            }
            predictionList.add(predictions);
        }
        errorAnalysis(new File(foldFolder, AuthorErrorFile),
                testDebateIndices,
                debateVoteData.getWords(),
                debateVoteData.getAuthors(),
                testAuthorIndices,
                testBillIndices,
                testVotes,
                modelNames,
                predictionList);
    }

    protected void analyzeError(File modelFolder) {
        File trResultFolder = new File(modelFolder, TRAIN_PREFIX + RESULT_FOLDER);
        File trPredictionFile = new File(trResultFolder, PREDICTION_FILE);
        if (trPredictionFile.exists()) {
            SparseVector[] predictions = AbstractVotePredictor.inputPredictions(trPredictionFile);
            ArrayList<String> modelNames = new ArrayList<>();
            modelNames.add(modelFolder.getName());
            ArrayList<SparseVector[]> predictionList = new ArrayList<>();
            predictionList.add(predictions);
            errorAnalysis(new File(trResultFolder, AuthorErrorFile),
                    trainDebateIndices,
                    debateVoteData.getWords(),
                    debateVoteData.getAuthors(),
                    trainAuthorIndices,
                    trainBillIndices,
                    trainVotes,
                    modelNames,
                    predictionList);
        }

        File teResultFolder = new File(modelFolder, TEST_PREFIX + RESULT_FOLDER);
        File tePredictionFile = new File(teResultFolder, PREDICTION_FILE);
        if (tePredictionFile.exists()) {
            SparseVector[] predictions = AbstractVotePredictor.inputPredictions(tePredictionFile);
            ArrayList<String> modelNames = new ArrayList<>();
            modelNames.add(modelFolder.getName());
            ArrayList<SparseVector[]> predictionList = new ArrayList<>();
            predictionList.add(predictions);
            errorAnalysis(new File(teResultFolder, AuthorErrorFile),
                    testDebateIndices,
                    debateVoteData.getWords(),
                    debateVoteData.getAuthors(),
                    testAuthorIndices,
                    testBillIndices,
                    testVotes,
                    modelNames,
                    predictionList);
        }
    }

    protected void errorAnalysis(File outputFile,
            ArrayList<Integer> debateIndices,
            int[][] words,
            int[] authors,
            ArrayList<Integer> authorIndices,
            ArrayList<Integer> billIndices,
            boolean[][] validVotes,
            ArrayList<String> modelNames,
            ArrayList<SparseVector[]> predictionList) {
        if (verbose) {
            logln("Analyzing errors " + outputFile);
        }
        if (billIndices == null) {
            billIndices = new ArrayList<>();
        }
        for (int bb = 0; bb < validVotes[0].length; bb++) {
            billIndices.add(bb);
        }

        SparseCount debateCounts = new SparseCount();
        SparseCount tokenCounts = new SparseCount();
        for (int dd : debateIndices) {
            int author = authors[dd];
            debateCounts.increment(author);
            tokenCounts.changeCount(author, words[dd].length);
        }

        try {
            BufferedWriter writer = IOUtils.getBufferedWriter(outputFile);
            writer.write("index"
                    + "\tauthor-index"
                    + "\tauthor-id"
                    + "\tfreedomworks-id"
                    + "\tname"
                    + "\tnum-with"
                    + "\tnum-against"
                    + "\tnum-total"
                    + "\tnum-debates"
                    + "\tnum-tokens"
                    + "\tparty"
                    + "\tnominate-score");
            for (String modelName : modelNames) {
                writer.write("\t" + modelName);
            }
            writer.write("\n");

            for (int aa = 0; aa < authorIndices.size(); aa++) {
                int author = authorIndices.get(aa);
                String authorId = debateVoteData.getAuthorVocab().get(author);

                int numTotal = 0;
                int numWith = 0;
                int numAgainst = 0;
                for (Integer billIndice : billIndices) {
                    int bill = billIndice;
                    if (validVotes[author][bill]) {
                        if (votes[author][bill] == Vote.WITH) {
                            numWith++;
                        } else if (votes[author][bill] == Vote.AGAINST) {
                            numAgainst++;
                        }
                        numTotal++;
                    }
                }
                writer.write(aa
                        + "\t" + author
                        + "\t" + authorId
                        + "\t" + debateVoteData.getAuthorProperty(authorId, GTLegislator.FW_ID)
                        + "\t" + debateVoteData.getAuthorProperty(authorId, GTLegislator.NAME)
                        + "\t" + numWith
                        + "\t" + numAgainst
                        + "\t" + numTotal
                        + "\t" + debateCounts.getCount(author)
                        + "\t" + tokenCounts.getCount(author)
                        + "\t" + debateVoteData.getAuthorProperty(authorId, GTLegislator.PARTY)
                        + "\t" + debateVoteData.getAuthorProperty(authorId, GTLegislator.NOMINATE_SCORE1));
                for (int ii = 0; ii < modelNames.size(); ii++) {
                    SparseVector[] predictions = predictionList.get(ii);
                    int numCorrect = 0;
                    for (Integer billIndice : billIndices) {
                        int bill = billIndice;
                        if (validVotes[author][bill]) {
                            if (!predictions[author].containsIndex(bill)) {
                                throw new RuntimeException("Empty prediction");
                            }

                            if (isCorrectlyPredicted(votes[author][bill], predictions[author].get(bill))) {
                                numCorrect++;
                            }
                        }
                    }
                    writer.write("\t" + numCorrect);
                }
                writer.write("\n");
            }
            writer.close();
        } catch (IOException e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while outputing to " + outputFile);
        }
    }

    private boolean isCorrectlyPredicted(int vote, double prob) {
        return (vote == Vote.WITH && prob >= 0.5) || (vote == Vote.AGAINST && prob < 0.5);
    }

    public static void addOptions() {
        // create the command line parser
        parser = new BasicParser();

        // create the Options
        options = new Options();

        // directories
        addOption("congress-num", "Congress number");
        addOption("data-folder", "Folder storing processed data");
        addOption("format-folder", "Format folder");
        addOption("model-folder", "Model folder");
        addOption("expt-folder", "Experiment folder");
        addOption("fw-folder", "FreedomWorks data folder");
        addOption("processed-data-folder", "Processed data folder");
        addOption("processed-bill-data-folder", "Processed bill data folder");

        addOption("tr2dev-ratio", "Training-to-developmeng ratio");
        addOption("te-ratio", "Test ratio");

        addOption("debate-file", "Debate formatted file name");
        addOption("bill-file", "Bill formatted file name");
        addOption("legislator-file", "Legislator file");

        // files with Tea Party annotation
        addOption("house-rep-file", "House Republcian file");
        addOption("senate-rep-file", "Senate Republcian file");

        addOption("run-mode", "Run mode");
        addOption("model", "Model");

        // mlr
        addOption("reg-type", "Regularization type");
        addOption("mlr-param", "Variance of parameters");
        addOption("agg-type", "Aggregate type");

        addOption("tau-sigma", "Variance");

        addOption("K", "Number of topics");
        addOption("J", "Number of frames");
        addOption("L", "Number of maximum levels");

        addOption("gem-mean", "Mean of GEM");
        addOption("gem-scale", "Scale of GEM");
        addOption("sigmas", "Vector of variances");
        addOption("mus", "Vector of means");
        addOption("alphas", "Alphas");
        addOption("betas", "Betas");
        addOption("topic-beta", "Topic beta");
        addOption("frame-beta", "Frame beta");
        addOption("pis", "Pi's");
        addOption("gammas", "Gamma's");
        addOption("gamma-means", "Gamma means");
        addOption("gamma-scales", "Gamma scales");
        addOption("global-alphas", "Global alphas");
        addOption("local-alphas", "Local alphas");
        addOption("global-alpha", "Global alpha");
        addOption("local-alpha", "Local alpha");
        addOption("rate-alpha", "Alpha for learning rate");
        addOption("rate-eta", "Eta for learning rate");
        addOption("wwt", "Word weight type");

        addOption("init-maxiter", "Init max iter");
        addOption("path", "Path assumption");

        // mode parameters
        addGreekParametersOptions();

        // processing options
        addCorpusProcessorOptions();

        // sampling
        addSamplingOptions();

        // cross validation
        addCrossValidationOptions();

        // liblinear
        addOption("C", "Cost of constraints violation in LibLinear");
        addOption("solver-type", "Type of the solver");
        addOption("feature-type", "Type of the feature (class or score)");

        // sa-nhdp
        addOption("beta-a", "Pseudo-count for STAY");
        addOption("beta-b", "Pseudo-count for PASS");

        addOption("opt-type", "Optimization type");
        addOption("norm-type", "Normalization type");
        addOption("params", "params");
        addOption("l1", "L1");
        addOption("l2", "L2");
        addOption("lexl1", "Lexical L1");
        addOption("lexl2", "Lexical L2");
        addOption("etal2", "eta's L2");
        options.addOption("up", false, "Uniform Process");
        options.addOption("mh", false, "mh");
        options.addOption("tfidf", false, "tfidf");

        options.addOption("roottopic", false, "roottopic");
        options.addOption("mh", false, "Metropolis-Hastings");
        options.addOption("initialize", false, "initialize");
        options.addOption("train", false, "train");
        options.addOption("dev", false, "development");
        options.addOption("test", false, "test");
        options.addOption("testvote", false, "Predict held out votes");
        options.addOption("testauthor", false, "Predict votes of held out authors");
        options.addOption("visualizeauthor", false, "Visualize authors");

        options.addOption("coherence", false, "coherence");
        options.addOption("parallel", false, "parallel");
        options.addOption("display", false, "display");
        options.addOption("visualize", false, "visualize");
        options.addOption("hack", false, "hack");
        options.addOption("analyze", false, "Analyze");
        options.addOption("analyzeerror", false, "Analyze prediction error");
        options.addOption("testeval", false, "Evaluate test predictions");

        // logistic regression features
        options.addOption("sldamult", false, "SLDA-Mult features");
        options.addOption("snlda", false, "SNLDA features");
        options.addOption("slda", false, "SLDA features");
        options.addOption("party", false, "Party features");
        options.addOption("lip", false, "Lexical Ideal Point");
        options.addOption("lexicalsnlda", false, "Party features");

        addOption("branch-factors", "Initial branching factors at each level. "
                + "The length of this array should be equal to L-1 (where L "
                + "is the number of levels in the tree).");

        options.addOption("html", false, "Outputing to html file");
        options.addOption("paramOpt", false, "Optimizing parameters");
        options.addOption("diagnose", false, "diagnose");
        options.addOption("v", false, "verbose");
        options.addOption("d", false, "debug");
        options.addOption("z", false, "z-normalize");
        options.addOption("help", false, "Help");
    }

    public static void main(String[] args) {
        try {
            long sTime = System.currentTimeMillis();

            addOptions();

            cmd = parser.parse(options, args);
            if (cmd.hasOption("help")) {
                CLIUtils.printHelp(getHelpString(VotePredExpt.class.getName()), options);
                return;
            }

            verbose = cmd.hasOption("v");
            debug = cmd.hasOption("d");

            Congress.setVerbose(verbose);
            Congress.setDebug(debug);
            TextDataset.setDebug(debug);
            TextDataset.setVerbose(verbose);

            VotePredExpt expt = new VotePredExpt();
            expt.setup();
            String runMode = CLIUtils.getStringArgument(cmd, "run-mode", "run");
            switch (runMode) {
                case "preprocess":
                    expt.preprocess();
                    break;
                case "preprocess-vote":
                    expt.preprocessVoteText();
                    break;
                case "run":
                    expt.run();
                    break;
                case "adhoc":
                    expt.adhocProcess();
                    break;
                case "hack":
                    expt.debug();
                    break;
                default:
                    throw new RuntimeException("Run mode " + runMode + " is not supported");
            }

            // date and time
            DateFormat df = new SimpleDateFormat("dd/MM/yy HH:mm:ss");
            Date dateobj = new Date();
            long eTime = (System.currentTimeMillis() - sTime) / 1000;
            System.out.println("Elapsed time: " + eTime + "s");
            System.out.println("End time: " + df.format(dateobj));
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    class Initializer {

        int[][] billWords;
        int[] billLabels;
        int[][] debateWords;

        int L;
        int V;

        double[][] labelPriors;
        double[] backgrounPriors;

        public Initializer(int[][] billWords, int[] billLabels,
                int[][] debateWords,
                int L, int V) {
            this.billWords = billWords;
            this.billLabels = billLabels;
            this.debateWords = debateWords;
            this.L = L;
            this.V = V;
        }

        void estimate() {
            // estimate empirical distributions from bill data
            if (verbose) {
                logln("--- Estimating from bill data ...");
            }
            int billD = this.billWords.length;
            SparseVector[] billVectors = new SparseVector[L];
            for (int ll = 0; ll < L; ll++) {
                billVectors[ll] = new SparseVector(V);
            }
            SparseVector billBackground = new SparseVector(V);
            for (int dd = 0; dd < billD; dd++) {
                for (int nn = 0; nn < billWords[dd].length; nn++) {
                    billBackground.change(billWords[dd][nn], 1.0);
                    billVectors[billLabels[dd]].change(billWords[dd][nn], 1.0);
                }
            }

            // normalize the "differences"
            billBackground.normalize();
            for (int ll = 0; ll < L; ll++) {
                billVectors[ll].normalize();
                for (int vv : billVectors[ll].getIndices()) {
                    double diff = billVectors[ll].get(vv) - billBackground.get(vv);
                    if (diff < 0) {
                        diff = 0;
                    }
                    billVectors[ll].set(vv, diff);
                }
                billVectors[ll].normalize();
            }

            labelPriors = new double[L][];
            for (int ll = 0; ll < L; ll++) {
                labelPriors[ll] = billVectors[ll].dense();
            }
        }
    }
}
