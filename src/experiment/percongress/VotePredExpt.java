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
import java.io.BufferedWriter;
import java.io.File;
import java.io.IOException;
import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Date;
import java.util.HashMap;
import java.util.HashSet;
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
import util.MiscUtils;
import util.RankingItem;
import util.RankingItemList;
import util.StatUtils;
import util.normalizer.MinMaxNormalizer;
import votepredictor.BayesianMultIdealPoint;
import votepredictor.BayesianMultIdealPointOWLQN;
import votepredictor.SNLDAMultIdealPoint;

/**
 *
 * @author vietan
 */
public class VotePredExpt extends AbstractExperiment<Congress> {

    public static final String GovtrackUrl = "https://www.govtrack.us/data/us/";
    public static final String AuthorScoreFile = "authors.score";
    public static final String VoteScoreFile = "votes.score";
    public static final String PRE_SCORE = "pre-score";
    public static final String POS_SCORE = "pos-score";
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
    // dev
    protected boolean[][] devVotes;
    // test
    protected ArrayList<Integer> testAuthorIndices;
    protected ArrayList<Integer> testBillIndices;
    protected ArrayList<Integer> testDebateIndices;
    protected boolean[][] testVotes;

    protected Set<String> keyvoteBills;

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

    @Override
    public void setup() {
        if (verbose) {
            logln("Setting up ...");
        }

        congressNum = CLIUtils.getStringArgument(cmd, "congress-num", "112");
        datasetFolder = CLIUtils.getStringArgument(cmd, "data-folder",
                "/fs/clip-political/vietan/data/govtrack");
        numTopWords = CLIUtils.getIntegerArgument(cmd, "num-top-words", 20);
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
            voteText.add(keyvote.getProperty(FWBill.TITLE) + " "
                    + keyvote.getProperty(FWBill.SUMMARY));
        }

        CorpusProcessor corpProc = TextDataset.createCorpusProcessor();
        TextDataset voteDataset = new TextDataset(congressNum, processedDataFolder, corpProc);
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
                    int rollcall = fwYears[ii].getKeyRollCalls().get(jj);
                    legVotes.put(years[ii] + "_" + rollcall, lv.get(jj));
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
                null, null);
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
            voteText.add(keyvote.getProperty(FWBill.TITLE) + " "
                    + keyvote.getProperty(FWBill.SUMMARY));
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
        legislatorProperties.add("type");
        legislatorProperties.add(GTLegislator.NOMINATE_SCORE1);
        legislatorProperties.add(GTLegislator.FW_SCORE);

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
    }

    protected void loadFormattedData() {
        if (verbose) {
            logln("--- Loading debate data from " + processedDataFolder);
        }
        debateVoteData = new AuthorVoteTextDataset(congressNum, processedDataFolder);
        debateVoteData.loadFormattedData(processedDataFolder);
        votes = debateVoteData.getVotes();

        if (verbose) {
            logln("--- Loading bill data from " + processedDataFolder);
        }
        billData = new Bill("bill-" + congressNum, processedDataFolder);
        billData.loadFormattedData(processedDataFolder);

        if (verbose) {
            logln("--- Data loaded.");
        }
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

    /**
     * Run a model.
     *
     * @param outputFolder Output folder
     */
    protected void runModel(File outputFolder) {
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
            case "slda-ideal-point":
                runSLDAIdealPoint(outputFolder);
                break;
            case "snlda-ideal-point":
                runSNLDAIdealPoint(outputFolder);
                break;
            case "snlda-mult-ideal-point":
                runSNLDAMultIdealPoint(outputFolder);
                break;
            case "snhdp-ideal-point":
                runSNHDPIdealPoint(outputFolder);
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
        sampler.train(trainDebateIndices,
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
                    AbstractVotePredictor.evaluate(votes, testVotes, predictions));
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
                    AbstractVotePredictor.evaluate(votes, testVotes, predictions));
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
        }

        if (cmd.hasOption("testvote")) {
            pred.input(new File(predFolder, MODEL_FILE));
            SparseVector[] predictions = pred.test(testVotes);
            File teResultFolder = new File(predFolder, TEST_PREFIX + RESULT_FOLDER);
            IOUtils.createFolder(teResultFolder);
            AbstractModel.outputPerformances(new File(teResultFolder, RESULT_FILE),
                    AbstractVotePredictor.evaluate(votes, testVotes, predictions));
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
                                    .evaluate(votes, testVotes, predictions);
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
        }

        if (cmd.hasOption("testvote")) {
            pred.input(new File(predFolder, MODEL_FILE));
            SparseVector[] predictions = pred.test(testVotes);
            File teResultFolder = new File(predFolder, TEST_PREFIX + RESULT_FOLDER);
            IOUtils.createFolder(teResultFolder);
            AbstractModel.outputPerformances(new File(teResultFolder, RESULT_FILE),
                    AbstractVotePredictor.evaluate(votes, testVotes, predictions));
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
                    AbstractVotePredictor.evaluate(votes, testVotes, predictions));
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
            sampler.train(trainDebateIndices,
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
        }

        if (cmd.hasOption("testvote")) { // average over multiple test chains
            SparseVector[] predictions = null;
            int count = 0;
            File reportFolder = new File(sampler.getReportFolderPath());
            String[] files = reportFolder.list();
            for (String file : files) {
                if (!file.endsWith(".zip")) {
                    continue;
                }
                sampler.inputState(new File(reportFolder, file));
                SparseVector[] partPreds = sampler.test(testVotes);
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
            File teResultFolder = new File(samplerFolder, TEST_PREFIX + RESULT_FOLDER);
            IOUtils.createFolder(teResultFolder);
            AbstractVotePredictor.outputPredictions(new File(teResultFolder, PREDICTION_FILE),
                    votes, predictions);
            AbstractModel.outputPerformances(new File(teResultFolder, RESULT_FILE),
                    AbstractVotePredictor.evaluate(votes, testVotes, predictions));
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
                // TODO: average author scores
            } else {
                predictions = sampler.test(null,
                        testDebateIndices,
                        debateVoteData.getWords(),
                        debateVoteData.getAuthors(),
                        testAuthorIndices,
                        testVotes,
                        null);
                outputAuthorScore(new File(teResultFolder, AuthorScoreFile),
                        debateVoteData.getAuthorVocab(),
                        testAuthorIndices,
                        testVotes,
                        sampler.getPredictedUs(),
                        debateVoteData.getAuthorTable());
            }

            String predFile = PREDICTION_FILE;
            String resultFile = RESULT_FILE;
            AbstractVotePredictor.outputPredictions(new File(teResultFolder, predFile),
                    votes, predictions);
            AbstractModel.outputPerformances(new File(teResultFolder, resultFile),
                    AbstractVotePredictor.evaluate(votes, testVotes, predictions));
        }

        if (cmd.hasOption("testauthorlogreg")) {
            File teResultFolder = new File(samplerFolder, TEST_PREFIX + RESULT_FOLDER);
            IOUtils.createFolder(teResultFolder);
        }
    }

    protected void runSNLDAIdealPoint(File outputFolder) {
        // estimate seeded issues
        double[][] issuePhis = estimateIssues();
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
        sampler.setLabelVocab(billData.getTopicVocab());

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

        if (cmd.hasOption("train")) {
            sampler.train(trainDebateIndices,
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

        if (cmd.hasOption("testvote")) { // average over multiple test chains
            SparseVector[] predictions = null;
            int count = 0;
            File reportFolder = new File(sampler.getReportFolderPath());
            String[] files = reportFolder.list();
            for (String file : files) {
                if (!file.endsWith(".zip")) {
                    continue;
                }
                sampler.inputState(new File(reportFolder, file));
                SparseVector[] partPreds = sampler.test(testVotes);
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
            File teResultFolder = new File(samplerFolder, TEST_PREFIX + RESULT_FOLDER);
            IOUtils.createFolder(teResultFolder);
            AbstractVotePredictor.outputPredictions(new File(teResultFolder, PREDICTION_FILE),
                    votes, predictions);
            AbstractModel.outputPerformances(new File(teResultFolder, RESULT_FILE),
                    AbstractVotePredictor.evaluate(votes, testVotes, predictions));
        }

        if (cmd.hasOption("testauthor")) {
            File teResultFolder = new File(samplerFolder, TEST_PREFIX + RESULT_FOLDER);
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

                // average author scores
                String[] filenames = iterPredFolderPath.list();
                double[] authorScores = null;
                int count = 0;
                for (String filename : filenames) {
                    if (!filename.contains(AbstractVotePredictor.AuthorScoreFile)) {
                        continue;
                    }
                    double[] partAuthorScores = AbstractVotePredictor.inputAuthorScores(
                            new File(iterPredFolderPath, filename));
                    if (authorScores == null) {
                        authorScores = partAuthorScores;
                    } else {
                        for (int aa = 0; aa < authorScores.length; aa++) {
                            authorScores[aa] += partAuthorScores[aa];
                        }
                    }
                    count++;
                }
                for (int aa = 0; aa < authorScores.length; aa++) {
                    authorScores[aa] /= count;
                }
                outputAuthorScore(new File(teResultFolder, AuthorScoreFile),
                        debateVoteData.getAuthorVocab(),
                        testAuthorIndices,
                        testVotes,
                        authorScores,
                        debateVoteData.getAuthorTable());

            } else {
                predictions = sampler.test(null,
                        testDebateIndices,
                        debateVoteData.getWords(),
                        debateVoteData.getAuthors(),
                        testAuthorIndices,
                        testVotes,
                        null, null, null,
                        new File(samplerFolder, TEST_PREFIX + "assignments.zip"));
                outputAuthorScore(new File(teResultFolder, AuthorScoreFile),
                        debateVoteData.getAuthorVocab(),
                        testAuthorIndices,
                        testVotes,
                        sampler.getPredictedUs(),
                        debateVoteData.getAuthorTable());
            }

            String predFile = PREDICTION_FILE;
            String resultFile = RESULT_FILE;
            AbstractVotePredictor.outputPredictions(new File(teResultFolder, predFile),
                    votes, predictions);
            AbstractModel.outputPerformances(new File(teResultFolder, resultFile),
                    AbstractVotePredictor.evaluate(votes, testVotes, predictions));
        }

        if (cmd.hasOption("visualize")) {
            sampler.train(trainDebateIndices,
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

        if (cmd.hasOption("visualizeauthor")) {
            sampler.train(trainDebateIndices,
                    debateVoteData.getWords(),
                    debateVoteData.getAuthors(),
                    votes,
                    trainAuthorIndices,
                    trainBillIndices,
                    trainVotes);
            sampler.inputFinalState();
            File authorTopicFile = new File(samplerFolder, "author-topic-dists.txt");
//            outputAuthorScore(authorTopicFile,
//                    debateVoteData.getAuthorVocab(),
//                    trainAuthorIndices,
//                    trainVotes,
//                    sampler.getUs(),
//                    debateVoteData.getAuthorTable(),
//                    sampler.getAuthorTopicDistributions(),
//                    sampler.getTopicScores(),
//                    billData.getTopicVocab());
        }
    }
    
    protected void runSNLDAMultIdealPoint(File outputFolder) {
        // estimate seeded issues
        double[][] issuePhis = estimateIssues();
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
        double[] sigmas = CLIUtils.getDoubleArrayArgument(cmd, "sigmas",
                new double[]{0.0, 1.0, 2.5}, ",");
        double sigma = CLIUtils.getDoubleArgument(cmd, "sigma", 2.5);
        double l1 = CLIUtils.getDoubleArgument(cmd, "l1", 0.0);
        double l2 = CLIUtils.getDoubleArgument(cmd, "l2", 1.0);
        boolean hasRootTopic = cmd.hasOption("roottopic");

        SNLDAMultIdealPoint sampler = new SNLDAMultIdealPoint();
        sampler.setVerbose(verbose);
        sampler.setDebug(debug);
        sampler.setLog(true);
        sampler.setReport(true);
        sampler.setWordVocab(debateVoteData.getWordVocab());
        sampler.setAuthorVocab(debateVoteData.getAuthorVocab());
        sampler.setLabelVocab(billData.getTopicVocab());

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
                mu, sigmas, sigma, l1, l2, hasRootTopic,
                initState, pathAssumption, paramOpt,
                burn_in, max_iters, sample_lag, report_interval);
        File samplerFolder = new File(outputFolder, sampler.getSamplerFolder());
        IOUtils.createFolder(samplerFolder);

        if (cmd.hasOption("train")) {
            sampler.setupData(trainDebateIndices,
                    debateVoteData.getWords(),
                    debateVoteData.getAuthors(),
                    votes, trainAuthorIndices, trainBillIndices,
                    trainVotes);
            sampler.initialize();
            sampler.iterate();
            sampler.outputTopicTopWords(new File(samplerFolder, TopWordFile), numTopWords);
        }

        if (cmd.hasOption("testvote")) { // average over multiple test chains
//            SparseVector[] predictions = null;
//            int count = 0;
//            File reportFolder = new File(sampler.getReportFolderPath());
//            String[] files = reportFolder.list();
//            for (String file : files) {
//                if (!file.endsWith(".zip")) {
//                    continue;
//                }
//                sampler.inputState(new File(reportFolder, file));
//                SparseVector[] partPreds = sampler.test(testVotes);
//                if (predictions == null) {
//                    predictions = partPreds;
//                } else {
//                    for (int aa = 0; aa < predictions.length; aa++) {
//                        predictions[aa].add(partPreds[aa]);
//                    }
//                }
//                count++;
//            }
//            for (SparseVector prediction : predictions) {
//                prediction.scale(1.0 / count);
//            }
//            File teResultFolder = new File(samplerFolder, TEST_PREFIX + RESULT_FOLDER);
//            IOUtils.createFolder(teResultFolder);
//            AbstractVotePredictor.outputPredictions(new File(teResultFolder, PREDICTION_FILE),
//                    votes, predictions);
//            AbstractModel.outputPerformances(new File(teResultFolder, RESULT_FILE),
//                    AbstractVotePredictor.evaluate(votes, testVotes, predictions));
        }

        if (cmd.hasOption("testauthor")) {
            File teResultFolder = new File(samplerFolder, TEST_PREFIX + RESULT_FOLDER);
            IOUtils.createFolder(teResultFolder);

            SparseVector[] predictions = null;
            if (cmd.hasOption("parallel")) {
//                File iterPredFolderPath = new File(samplerFolder,
//                        AbstractSampler.IterPredictionFolder);
//                predictions = SNLDAIdealPoint.parallelTest(testDebateIndices,
//                        debateVoteData.getWords(),
//                        debateVoteData.getAuthors(),
//                        testAuthorIndices,
//                        testVotes,
//                        iterPredFolderPath,
//                        sampler);
//
//                // average author scores
//                String[] filenames = iterPredFolderPath.list();
//                double[] authorScores = null;
//                int count = 0;
//                for (String filename : filenames) {
//                    if (!filename.contains(AbstractVotePredictor.AuthorScoreFile)) {
//                        continue;
//                    }
//                    double[] partAuthorScores = AbstractVotePredictor.inputAuthorScores(
//                            new File(iterPredFolderPath, filename));
//                    if (authorScores == null) {
//                        authorScores = partAuthorScores;
//                    } else {
//                        for (int aa = 0; aa < authorScores.length; aa++) {
//                            authorScores[aa] += partAuthorScores[aa];
//                        }
//                    }
//                    count++;
//                }
//                for (int aa = 0; aa < authorScores.length; aa++) {
//                    authorScores[aa] /= count;
//                }
//                outputAuthorScore(new File(teResultFolder, AuthorScoreFile),
//                        debateVoteData.getAuthorVocab(),
//                        testAuthorIndices,
//                        testVotes,
//                        authorScores,
//                        debateVoteData.getAuthorTable());

            } else {
                predictions = sampler.test(null,
                        testDebateIndices,
                        debateVoteData.getWords(),
                        debateVoteData.getAuthors(),
                        testAuthorIndices,
                        testVotes,
                        null, null, null,
                        new File(samplerFolder, TEST_PREFIX + "assignments.zip"));
                
//                predictions = sampler.test(null,
//                        testDebateIndices,
//                        debateVoteData.getWords(),
//                        debateVoteData.getAuthors(),
//                        testAuthorIndices,
//                        testVotes,
//                        null, null, null);
                
//                outputAuthorScore(new File(teResultFolder, AuthorScoreFile),
//                        debateVoteData.getAuthorVocab(),
//                        testAuthorIndices,
//                        testVotes,
//                        sampler.getPredictedUs(),
//                        debateVoteData.getAuthorTable());
            }

            String predFile = PREDICTION_FILE;
            String resultFile = RESULT_FILE;
            AbstractVotePredictor.outputPredictions(new File(teResultFolder, predFile),
                    votes, predictions);
            AbstractModel.outputPerformances(new File(teResultFolder, resultFile),
                    AbstractVotePredictor.evaluate(votes, testVotes, predictions));
        }
    }

    protected void runSNHDPIdealPoint(File outputFolder) {
        // estimate seeded issues
        double[][] issuePhis = estimateIssues();
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
        sampler.setLabelVocab(billData.getTopicVocab());

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

        if (cmd.hasOption("train")) {
            sampler.train(trainDebateIndices,
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
            SparseVector[] predictions = null;
            int count = 0;
            File reportFolder = new File(sampler.getReportFolderPath());
            String[] files = reportFolder.list();
            for (String file : files) {
                if (!file.endsWith(".zip")) {
                    continue;
                }
                sampler.inputState(new File(reportFolder, file));
                SparseVector[] partPreds = sampler.test(testVotes);
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
            File teResultFolder = new File(samplerFolder, TEST_PREFIX + RESULT_FOLDER);
            IOUtils.createFolder(teResultFolder);
            AbstractVotePredictor.outputPredictions(new File(teResultFolder, PREDICTION_FILE),
                    votes, predictions);
            AbstractModel.outputPerformances(new File(teResultFolder, RESULT_FILE),
                    AbstractVotePredictor.evaluate(votes, testVotes, predictions));
        }

        if (cmd.hasOption("testauthor")) {
            File teResultFolder = new File(samplerFolder, TEST_PREFIX + RESULT_FOLDER);
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

                // average author scores
                String[] filenames = iterPredFolderPath.list();
                double[] authorScores = null;
                int count = 0;
                for (String filename : filenames) {
                    if (!filename.contains(AbstractVotePredictor.AuthorScoreFile)) {
                        continue;
                    }
                    double[] partAuthorScores = AbstractVotePredictor.inputAuthorScores(
                            new File(iterPredFolderPath, filename));
                    if (authorScores == null) {
                        authorScores = partAuthorScores;
                    } else {
                        for (int aa = 0; aa < authorScores.length; aa++) {
                            authorScores[aa] += partAuthorScores[aa];
                        }
                    }
                    count++;
                }
                for (int aa = 0; aa < authorScores.length; aa++) {
                    authorScores[aa] /= count;
                }
                outputAuthorScore(new File(teResultFolder, AuthorScoreFile),
                        debateVoteData.getAuthorVocab(),
                        testAuthorIndices,
                        testVotes,
                        authorScores,
                        debateVoteData.getAuthorTable());

            } else {
                predictions = sampler.test(null,
                        testDebateIndices,
                        debateVoteData.getWords(),
                        debateVoteData.getAuthors(),
                        testAuthorIndices,
                        testVotes,
                        null, null, null);
                outputAuthorScore(new File(teResultFolder, AuthorScoreFile),
                        debateVoteData.getAuthorVocab(),
                        testAuthorIndices,
                        testVotes,
                        sampler.getPredictedUs(),
                        debateVoteData.getAuthorTable());
            }

            String predFile = PREDICTION_FILE;
            String resultFile = RESULT_FILE;
            AbstractVotePredictor.outputPredictions(new File(teResultFolder, predFile),
                    votes, predictions);
            AbstractModel.outputPerformances(new File(teResultFolder, resultFile),
                    AbstractVotePredictor.evaluate(votes, testVotes, predictions));
        }
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
        if (verbose) {
            displayTopics(billData.getTopicVocab(), debateVoteData.getWordVocab(), priors);
        }

        // to avoid 0.0 probabilities
        for (double[] prior : priors) {
            for (int vv = 0; vv < prior.length; vv++) {
                prior[vv] = (prior[vv] + 1.0 / V) / 2;
            }
        }

        return priors;
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
                System.out.println(labelVocab.get(ll));
            } else {
                System.out.println("Topic " + ll);
            }

            for (int ii = 0; ii < 20; ii++) {
                RankingItem<Integer> item = rankWords.get(ii);
                System.out.print("\t" + wordVocab.get(item.getObject())
                        + ":" + MiscUtils.formatDouble(item.getPrimaryValue()));
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
                    + "\tFreedomWorksID\tName\tParty\tNominateScore\n");
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
                    writer.write("\t" + author.getProperty(GTLegislator.FW_ID)
                            + "\t" + author.getProperty(GTLegislator.NAME)
                            + "\t" + author.getProperty(GTLegislator.PARTY)
                            + "\t" + author.getProperty(GTLegislator.NOMINATE_SCORE1)
                    );
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
//            writer.write("Index\tID\tScore\tNumWithVotes\tNumAgainstVotes");
//            if (authorTable != null) {
//                writer.write("\tFreedomWorksID\tName\tParty\tNominateScore");
//            }
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
//                writer.write(aa + "\t" + aid
//                        + "\t" + authorScores[ii]
//                        + "\t" + withCount
//                        + "\t" + againstCount);
//                if (authorTable != null) {
//                    Author author = authorTable.get(aid);
//                    writer.write("\t" + author.getProperty(GTLegislator.FW_ID)
//                            + "\t" + author.getProperty(GTLegislator.NAME)
//                            + "\t" + author.getProperty(GTLegislator.PARTY)
//                            + "\t" + author.getProperty(GTLegislator.NOMINATE_SCORE1)
//                    );
//                }

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
        addOption("gamma-means", "Gamma means");
        addOption("gamma-scales", "Gamma scales");
        addOption("global-alphas", "Global alphas");
        addOption("local-alphas", "Local alphas");
        addOption("rate-alpha", "Alpha for learning rate");
        addOption("rate-eta", "Eta for learning rate");

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

        options.addOption("roottopic", false, "roottopic");
        options.addOption("mh", false, "Metropolis-Hastings");
        options.addOption("train", false, "train");
        options.addOption("dev", false, "development");
        options.addOption("test", false, "test");
        options.addOption("testvote", false, "Predict held out votes");
        options.addOption("testauthor", false, "Predict votes of held out authors");
        options.addOption("visualizeauthor", false, "Visualize authors");

        options.addOption("parallel", false, "parallel");
        options.addOption("display", false, "display");
        options.addOption("visualize", false, "visualize");
        options.addOption("hack", false, "hack");

        // logistic regression features
        options.addOption("snlda", false, "SNLDA features");
        options.addOption("party", false, "Party features");

        addOption("branch-factors", "Initial branching factors at each level. "
                + "The length of this array should be equal to L-1 (where L "
                + "is the number of levels in the tree).");

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
