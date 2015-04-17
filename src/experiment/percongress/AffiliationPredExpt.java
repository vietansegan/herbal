package experiment.percongress;

import cc.mallet.optimize.LimitedMemoryBFGS;
import core.AbstractSampler;
import sampling.likelihood.CascadeDirMult.PathAssumption;
import core.crossvalidation.CrossValidation;
import core.crossvalidation.Fold;
import core.crossvalidation.Instance;
import data.Congress;
import data.TextDataset;
import data.Vote;
import de.bwaldvogel.liblinear.Feature;
import de.bwaldvogel.liblinear.FeatureNode;
import de.bwaldvogel.liblinear.Linear;
import de.bwaldvogel.liblinear.Model;
import de.bwaldvogel.liblinear.Parameter;
import de.bwaldvogel.liblinear.Problem;
import de.bwaldvogel.liblinear.SolverType;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Date;
import java.util.HashMap;
import java.util.List;
import java.util.Random;
import optimization.OWLQNLogisticRegression;
import optimization.RidgeLogisticRegressionOptimizable;
import org.apache.commons.cli.ParseException;
import svm.SVMLight;
import svm.SVMUtils;
import util.CLIUtils;
import util.IOUtils;
import util.MiscUtils;
import util.MismatchRuntimeException;
import util.PredictionUtils;
import util.RankingItem;
import util.RankingItemList;
import util.SamplerUtils;
import util.SparseVector;
import util.StatUtils;
import util.evaluation.Measurement;
import util.evaluation.RankingPerformance;
import util.govtrack.GTLegislator;
import util.normalizer.MinMaxNormalizer;
import votepredictor.textidealpoint.hierarchy.HierMultSHDP;

/**
 *
 * @author vietan
 */
public class AffiliationPredExpt extends VotePredExpt {

    public static final String DATAFILE = "data";

    public enum AffiliationType {

        TPCAUCUS, TPEXPRESS, FREEDOMWORKS, SARAHPALIN, ANY, ALL
    }
    private AffiliationType affType;
    private ArrayList<Instance<Integer>> gopAuthorList;
    private ArrayList<Integer> gopTPAffList;

    // cross validation
    private String[] trAuthorIds;
    private int[] trAuthorLabels;
    private String[] teAuthorIds;
    private int[] teAuthorLabels;

    // internal
    private HashMap<Integer, ArrayList<Integer>> authorDocIndices;

    public AffiliationPredExpt(String affType) {
        switch (affType) {
            case "tpcaucus":
                this.affType = AffiliationType.TPCAUCUS;
                break;
            case "tpexpress":
                this.affType = AffiliationType.TPEXPRESS;
                break;
            case "freedomworks":
                this.affType = AffiliationType.FREEDOMWORKS;
                break;
            case "sarapalin":
                this.affType = AffiliationType.SARAHPALIN;
                break;
            case "all":
                this.affType = AffiliationType.ALL;
                break;
            case "any":
                this.affType = AffiliationType.ANY;
                break;
            default:
                throw new RuntimeException("Affiliation type " + affType + " not suppoted");
        }
    }

    @Override
    public String getConfiguredExptFolder() {
        return affType + "-cv-" + numFolds + "-" + teRatio + "-" + trToDevRatio;
    }

    public String getCVFolder() {
        return "cv-" + numFolds + "-" + teRatio + "-" + trToDevRatio;
    }

    @Override
    public void run() {
        if (verbose) {
            logln("Running ...");
        }
        try {
            setupSampling();
            loadFormattedData();
            runCrossValidation();
            evaluate();
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while running");
        }
    }

    @Override
    protected void loadFormattedData() {
        super.loadFormattedData();
        debateVoteData.computeTFIDFs();
        getGOPs();

        // indices of documents per author
        authorDocIndices = new HashMap<>();
        for (int dd = 0; dd < debateVoteData.getWords().length; dd++) {
            int aa = debateVoteData.getAuthors()[dd];
            ArrayList<Integer> docIndices = authorDocIndices.get(aa);
            if (docIndices == null) {
                docIndices = new ArrayList<>();
            }
            docIndices.add(dd);
            authorDocIndices.put(aa, docIndices);
        }

        if (verbose) {
            logln("--- # authors with text: " + authorDocIndices.size());
        }
    }

    private void runCrossValidation() throws Exception {
        if (verbose) {
            logln("--- Running cross validation ...");
        }

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

        File congressFolder = new File(experimentPath, congressNum);
        File configureFolder = new File(congressFolder, getConfiguredExptFolder());
        IOUtils.createFolder(configureFolder);

        ArrayList<Integer> runningFolds = new ArrayList<Integer>();
        if (cmd.hasOption("fold")) {
            String foldList = cmd.getOptionValue("fold");
            for (String f : foldList.split(",")) {
                runningFolds.add(Integer.parseInt(f));
            }
        }
        for (int ff = 0; ff < numFolds; ff++) {
            if (!runningFolds.isEmpty() && !runningFolds.contains(ff)) {
                continue;
            }
            if (verbose) {
                System.out.println("\nRunning fold " + ff);
            }

            Fold fold = new Fold(ff, new File(processedDataFolder, getCVFolder()).getAbsolutePath());
            inputFold(fold);

            File foldFeatureFolder = new File(new File(congressFolder, getCVFolder()), fold.getFoldName());
            File foldResultFolder = new File(configureFolder, fold.getFoldName());
            IOUtils.createFolder(foldResultFolder);

            String model = CLIUtils.getStringArgument(cmd, "model", "random");
            switch (model) {
                case "random":
                    runRandom(foldResultFolder);
                    break;
                case "test":
                    runTest(foldResultFolder);
                    break;
                case "vote":
                    getVoteFeatures(foldFeatureFolder);
                    runFeatures(foldFeatureFolder, foldResultFolder, "vote");
                    break;
                case "tf":
                    getTFFeatures(foldFeatureFolder);
                    runFeatures(foldFeatureFolder, foldResultFolder, "tf");
                    break;
                case "tfidf":
                    getTFIDFFeatures(foldResultFolder);
                    runFeatures(foldFeatureFolder, foldResultFolder, "tfidf");
                    break;
                case "hier-mult-shdp":
                    runHierMuilSHDP(foldFeatureFolder);
                    break;
                case "hiptm_all":
                    runFeatures(foldFeatureFolder, foldResultFolder, "Hier-Mult-SHDP_all");
                    break;
                case "combine":
                    runMetaCombinedFeature(foldFeatureFolder, foldResultFolder);
                    break;
                default:
                    throw new RuntimeException("Model " + model + " not supported");
            }
        }
    }

    private boolean isNormalizing() {
        return cmd.hasOption("normalize");
    }

    private void runClassifier(File folder, String modelName,
            SparseVector[] trainFeatures, SparseVector[] testFeatures,
            ArrayList<String> featureNames) throws Exception {
        String classifier = CLIUtils.getStringArgument(cmd, "classifier", "owlqn");
        switch (classifier) {
            case "owlqn":
                runLogisticRegressionOWLQN(folder, modelName, trainFeatures, testFeatures, featureNames);
                break;
            case "liblinear":
                runLibLinear(folder, modelName, trainFeatures, testFeatures);
                break;
            case "svmlight":
                if (affType == AffiliationType.ALL) {
                    runSVMRank(folder, modelName, trainFeatures, testFeatures);
                } else {
                    runSVMLightClassification(folder, modelName, trainFeatures, testFeatures);
                }
                break;
            default:
                throw new RuntimeException("Classifier " + classifier + " not supported");
        }
    }

    protected void runHierMuilSHDP(File foldResultFolder) throws Exception {
        double[][] billTopicPriors = createBillPriors();
        int K;
        double[][] issuePhis;
        if (cmd.hasOption("K")) {
            issuePhis = null;
            K = Integer.parseInt(cmd.getOptionValue("K"));
        } else {
            issuePhis = estimateIssues();
            K = issuePhis.length;
        }
        int threshold = CLIUtils.getIntegerArgument(cmd, "T", 25);
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

        sampler.configure(foldResultFolder.getAbsolutePath(),
                debateVoteData.getWordVocab().size(), K, J,
                topicAlpha, frameAlphaGlobal, frameAlphaLocal, topicBeta, frameBeta,
                rho, sigma, gamma, lambda, epsilon, threshold,
                pathAssumption, issuePhis,
                initState, paramOpt,
                burn_in, max_iters, sample_lag, report_interval);
        File samplerFolder = new File(foldResultFolder, sampler.getSamplerFolder());
        IOUtils.createFolder(samplerFolder);

        if (cmd.hasOption("sampletrain")) {
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

            sampler.outputAuthorDetails(debateVoteData.getAuthorTable());

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

            outputVoteScores(new File(samplerFolder, VoteScoreFile),
                    debateVoteData.getVoteVocab(),
                    trainBillIndices,
                    sampler.getXs(), sampler.getYs(),
                    debateVoteData.getVoteTable());
        }

        if (cmd.hasOption("sampletest")) {
            File testReportFolder = new File(sampler.getSamplerFolderPath(),
                    TEST_PREFIX + AbstractSampler.ReportFolder);

            if (cmd.hasOption("parallel")) {
                HierMultSHDP.parallelTest(testDebateIndices,
                        debateVoteData.getWords(),
                        debateVoteData.getAuthors(),
                        null,
                        testAuthorIndices,
                        testVotes,
                        new File(samplerFolder, AbstractSampler.IterPredictionFolder),
                        testReportFolder,
                        sampler);
            } else {
                sampler.setAuthorVocab(debateVoteData.getAuthorVocab());
                sampler.setupData(testDebateIndices,
                        debateVoteData.getWords(),
                        debateVoteData.getAuthors(),
                        null,
                        testAuthorIndices,
                        testBillIndices,
                        testVotes);
                sampler.test(null, null, testReportFolder);
            }
        }

        if (cmd.hasOption("sampleall")) {
            File congressFolder = new File(experimentPath, congressNum);
            sampler.setFolder(new File(congressFolder, getCVFolder()).getAbsolutePath());

            ArrayList<Integer> allAuthorIndices = new ArrayList<>();
            for (int aa : trainAuthorIndices) {
                allAuthorIndices.add(aa);
            }
            for (int aa : testAuthorIndices) {
                allAuthorIndices.add(aa);
            }
            ArrayList<Integer> allDebateIndices = new ArrayList<>();
            for (int dd = 0; dd < debateVoteData.getWords().length; dd++) {
                int aa = debateVoteData.getAuthors()[dd];
                if (allAuthorIndices.contains(aa)) {
                    allDebateIndices.add(dd);
                }
            }

            boolean[][] allVotes = new boolean[votes.length][];
            for (int aa = 0; aa < votes.length; aa++) {
                allVotes[aa] = new boolean[votes[aa].length];
                Arrays.fill(allVotes[aa], false);
                if (allAuthorIndices.contains(aa)) {
                    for (int bb = 0; bb < votes[aa].length; bb++) {
                        if (votes[aa][bb] != Vote.MISSING) {
                            allVotes[aa][bb] = true;
                        }
                    }
                }
            }

            sampler.setupData(allDebateIndices,
                    debateVoteData.getWords(),
                    debateVoteData.getAuthors(),
                    votes,
                    allAuthorIndices,
                    trainBillIndices,
                    allVotes);
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
                    allVotes,
                    sampler.getUs(),
                    sampler.getMultiUs(),
                    debateVoteData.getAuthorTable());

            outputVoteScores(new File(samplerFolder, VoteScoreFile + ".top-topics.init"),
                    debateVoteData.getVoteVocab(),
                    trainBillIndices,
                    sampler.getBillThetas(),
                    debateVoteData.getVoteTable(), 5);

            sampler.metaIterate();

            sampler.outputAuthorDetails(debateVoteData.getAuthorTable());

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
                    allVotes,
                    sampler.getUs(),
                    sampler.getMultiUs(),
                    debateVoteData.getAuthorTable());

            outputVoteScores(new File(samplerFolder, VoteScoreFile),
                    debateVoteData.getVoteVocab(),
                    trainBillIndices,
                    sampler.getXs(), sampler.getYs(),
                    debateVoteData.getVoteTable());
        }

        if (cmd.hasOption("sampleallfeature")) {
            File congressFolder = new File(experimentPath, congressNum);
            sampler.setFolder(new File(congressFolder, getCVFolder()).getAbsolutePath());

            ArrayList<Integer> allAuthorIndices = new ArrayList<>();
            for (int aa : trainAuthorIndices) {
                allAuthorIndices.add(aa);
            }
            for (int aa : testAuthorIndices) {
                allAuthorIndices.add(aa);
            }
            ArrayList<Integer> allDebateIndices = new ArrayList<>();
            for (int dd = 0; dd < debateVoteData.getWords().length; dd++) {
                int aa = debateVoteData.getAuthors()[dd];
                if (allAuthorIndices.contains(aa)) {
                    allDebateIndices.add(dd);
                }
            }

            boolean[][] allVotes = new boolean[votes.length][];
            for (int aa = 0; aa < votes.length; aa++) {
                allVotes[aa] = new boolean[votes[aa].length];
                Arrays.fill(allVotes[aa], false);
                if (allAuthorIndices.contains(aa)) {
                    for (int bb = 0; bb < votes[aa].length; bb++) {
                        if (votes[aa][bb] != Vote.MISSING) {
                            allVotes[aa][bb] = true;
                        }
                    }
                }
            }

            sampler.setupData(allDebateIndices,
                    debateVoteData.getWords(),
                    debateVoteData.getAuthors(),
                    votes,
                    allAuthorIndices,
                    trainBillIndices,
                    allVotes);

            File samplerReportFolder = new File(sampler.getReportFolderPath());
            SparseVector[] allFeatures = sampler.getAuthorFeatures(samplerReportFolder, samplerReportFolder);
            System.out.println("# all features: " + allFeatures.length);

            File shortSamplerFolder = new File(foldResultFolder, sampler.getBasename() + "_all");
            IOUtils.createFolder(shortSamplerFolder);

            SparseVector[] trainFeatures = new SparseVector[trainAuthorIndices.size()];
            for (int ii = 0; ii < trainFeatures.length; ii++) {
                int aa = trainAuthorIndices.get(ii);
                int aidx = allAuthorIndices.indexOf(aa);
                if (aidx < 0) {
                    throw new RuntimeException();
                }
                trainFeatures[ii] = allFeatures[aidx];
            }
            outputFeatures(new File(shortSamplerFolder, DATAFILE + TRAIN_POSFIX), trainFeatures);

            SparseVector[] testFeatures = new SparseVector[testAuthorIndices.size()];
            for (int ii = 0; ii < testFeatures.length; ii++) {
                int aa = testAuthorIndices.get(ii);
                int aidx = allAuthorIndices.indexOf(aa);
                if (aidx < 0) {
                    throw new RuntimeException();
                }
                testFeatures[ii] = allFeatures[aidx];
            }
            outputFeatures(new File(shortSamplerFolder, DATAFILE + TEST_POSFIX), testFeatures);
        }

        File shortSamplerFolder = new File(foldResultFolder, sampler.getBasename());
        IOUtils.createFolder(shortSamplerFolder);

        if (isTraining()) {
            sampler.setAuthorVocab(debateVoteData.getAuthorVocab());
            sampler.setupData(trainDebateIndices,
                    debateVoteData.getWords(),
                    debateVoteData.getAuthors(),
                    votes,
                    trainAuthorIndices,
                    trainBillIndices,
                    trainVotes);
            File samplerReportFolder = new File(sampler.getReportFolderPath());
            SparseVector[] trainFeatures = sampler.getAuthorFeatures(samplerReportFolder, samplerReportFolder);
            outputFeatures(new File(shortSamplerFolder, DATAFILE + TRAIN_POSFIX), trainFeatures);
        }

        if (isTesting()) {
            sampler.setAuthorVocab(debateVoteData.getAuthorVocab());
            sampler.setupData(testDebateIndices,
                    debateVoteData.getWords(),
                    debateVoteData.getAuthors(),
                    null,
                    testAuthorIndices,
                    testBillIndices,
                    testVotes);
            File samplerReportFolder = new File(sampler.getReportFolderPath());
            File testReportFolder = new File(sampler.getSamplerFolderPath(), TEST_PREFIX + AbstractSampler.ReportFolder);
            SparseVector[] testFeatures = sampler.getAuthorFeatures(samplerReportFolder, testReportFolder);
            outputFeatures(new File(shortSamplerFolder, DATAFILE + TEST_POSFIX), testFeatures);
        }
    }

    /**
     * Combine different set of features.
     */
    public void runMetaCombinedFeature(File foldFeatureFolder, File foldResultFolder) throws Exception {
        ArrayList<String> features = new ArrayList<>();
        features.add(getVoteFeatures(foldFeatureFolder));
//        features.add(getTFFeatures(foldFeatureFolder));
        features.add(getTFIDFFeatures(foldFeatureFolder));
//        features.add(new HierMultSHDP().getBasename());
//        features.add(new HierMultSHDP().getBasename() + "_all");
        System.out.println("Features: " + features.toString());

        String featureStr = CLIUtils.getStringArgument(cmd, "features", "");
        ArrayList<String> selectedFeatures = new ArrayList<>();
        if (!featureStr.trim().isEmpty()) {
            String[] ssl = featureStr.split(",");
            selectedFeatures.addAll(Arrays.asList(ssl));
        } else {
            selectedFeatures = features;
        }
        System.out.println("Selected features: " + selectedFeatures.toString());

        ArrayList<ArrayList<String>> featurePowerSet = MiscUtils.getPowerSet(selectedFeatures);
        for (ArrayList<String> featureSet : featurePowerSet) {
            if (featureSet.isEmpty()) {
                continue;
            }
            String name = featureSet.get(0);
            for (int ii = 1; ii < featureSet.size(); ii++) {
                name += "_" + featureSet.get(ii);
            }

            if (verbose) {
                logln(">>> >>> Running " + name + ". " + featureSet);
            }
            ArrayList<String> featureFolders = new ArrayList<>();
            for (String feature : featureSet) {
                featureFolders.add(new File(foldFeatureFolder, feature).getAbsolutePath());
            }
            runCombinedFeatures(foldResultFolder, name, featureFolders, null);
        }
    }

    /**
     * Run classifier using a given set of features.
     *
     * @param foldResultFolder
     * @param combinedName
     * @param featureFolders
     * @param normalizeFlags
     */
    public void runCombinedFeatures(File foldResultFolder, String combinedName,
            ArrayList<String> featureFolders, ArrayList<Boolean> normalizeFlags) throws Exception {
        SparseVector[] combinedTrainFeatures = null;
        SparseVector[] combinedTestFeatures = null;
        for (int ii = 0; ii < featureFolders.size(); ii++) {
            String featureFolder = featureFolders.get(ii);
            SparseVector[] trainFeatures = inputFeatures(new File(featureFolder, DATAFILE + TRAIN_POSFIX));
            MinMaxNormalizer[] norms = null;
            if (normalizeFlags != null && normalizeFlags.get(ii)) {
                norms = StatUtils.minmaxNormalizeTrainingData(trainFeatures, trainFeatures[0].getDimension());
            }

            SparseVector[] testFeatures = inputFeatures(new File(featureFolder, DATAFILE + TEST_POSFIX));
            if (norms != null) {
                StatUtils.normalizeTestData(testFeatures, norms);
            }

            if (combinedTrainFeatures == null) {
                combinedTrainFeatures = trainFeatures;
                combinedTestFeatures = testFeatures;
            } else {
                assert combinedTestFeatures != null;
                for (int aa = 0; aa < combinedTrainFeatures.length; aa++) {
                    combinedTrainFeatures[aa].concatenate(trainFeatures[aa]);
                }
                for (int aa = 0; aa < combinedTestFeatures.length; aa++) {
                    combinedTestFeatures[aa].concatenate(testFeatures[aa]);
                }
            }
        }

        runClassifier(foldResultFolder, combinedName, combinedTrainFeatures, combinedTestFeatures, null);
    }

    /**
     * Classify using raw votes.
     */
    protected String getVoteFeatures(File foldResultFolder) throws Exception {
        String modelName = "vote";
        File vFolder = new File(foldResultFolder, modelName);
        IOUtils.createFolder(vFolder);

        int B = debateVoteData.getVoteVocab().size();
        int trA = trAuthorIds.length;
        SparseVector[] trainFeatures = new SparseVector[trA];
        for (int ii = 0; ii < trA; ii++) {
            int aa = trainAuthorIndices.get(ii);
            trainFeatures[ii] = new SparseVector(B);
            for (int bb = 0; bb < B; bb++) {
                if (debateVoteData.getVotes()[aa][bb] == Vote.WITH) {
                    trainFeatures[ii].set(bb, 1.0);
                } else if (debateVoteData.getVotes()[aa][bb] == Vote.AGAINST) {
                    trainFeatures[ii].set(bb, -1.0);
                }
            }
        }
        outputFeatures(new File(vFolder, DATAFILE + TRAIN_POSFIX), trainFeatures);

        int teA = teAuthorIds.length;
        SparseVector[] testFeatures = new SparseVector[teA];
        for (int ii = 0; ii < teA; ii++) {
            int aa = testAuthorIndices.get(ii);
            testFeatures[ii] = new SparseVector(B);
            for (int bb = 0; bb < B; bb++) {
                if (debateVoteData.getVotes()[aa][bb] == Vote.WITH) {
                    testFeatures[ii].set(bb, 1.0);
                } else if (debateVoteData.getVotes()[aa][bb] == Vote.AGAINST) {
                    testFeatures[ii].set(bb, -1.0);
                }
            }
        }
        outputFeatures(new File(vFolder, DATAFILE + TEST_POSFIX), testFeatures);
        return modelName;
    }

    protected void runFeatures(File foldFeatureFolder, File foldResultFolder,
            String featureName) throws Exception {
        if (verbose) {
            logln("Running feature: " + featureName);
            logln("Feature fold: " + foldFeatureFolder);
            logln("Result folder: " + foldResultFolder);
        }

        File featureFolder = new File(foldFeatureFolder, featureName);
        SparseVector[] trainFeatures = inputFeatures(new File(featureFolder, DATAFILE + TRAIN_POSFIX));
        MinMaxNormalizer[] norms = null;
        if (isNormalizing()) {
            norms = StatUtils.minmaxNormalizeTrainingData(trainFeatures, trainFeatures[0].getDimension());
            featureName += "_normalized";
        }
        SparseVector[] testFeatures = inputFeatures(new File(featureFolder, DATAFILE + TEST_POSFIX));
        if (norms != null) {
            StatUtils.normalizeTestData(testFeatures, norms);
        }
        runClassifier(foldResultFolder, featureName, trainFeatures, testFeatures, null);
    }

    /**
     * Classify using TF-IDF.
     */
    protected String getTFIDFFeatures(File foldResultFolder) throws Exception {
        String modelName = "tfidf";
        if (isNormalizing()) {
            modelName += "_normalized";
        }
        File tfidfFolder = new File(foldResultFolder, modelName);
        IOUtils.createFolder(tfidfFolder);

        int V = debateVoteData.getWordVocab().size();
        int[][] words = debateVoteData.getWords();

        int trA = trAuthorIds.length;
        SparseVector[] trainFeatures = new SparseVector[trA];
        for (int ii = 0; ii < trA; ii++) {
            int aa = trainAuthorIndices.get(ii);
            trainFeatures[ii] = new SparseVector(V);
            ArrayList<Integer> docIndices = authorDocIndices.get(aa);
            if (docIndices != null) {
                for (int dd : authorDocIndices.get(aa)) {
                    for (int nn = 0; nn < words[dd].length; nn++) {
                        trainFeatures[ii].change(words[dd][nn], 1.0);
                    }
                }
                for (int vv : trainFeatures[ii].getIndices()) {
                    double tf = trainFeatures[ii].get(vv);
                    double tfidf = Math.log(tf + 1) * debateVoteData.getIDFs()[vv];
                    trainFeatures[ii].set(vv, tfidf);
                }
                if (isNormalizing()) {
                    trainFeatures[ii].normalize();
                }
            }
        }
        outputFeatures(new File(tfidfFolder, DATAFILE + TRAIN_POSFIX), trainFeatures);

        int teA = teAuthorIds.length;
        SparseVector[] testFeatures = new SparseVector[teA];
        for (int ii = 0; ii < teA; ii++) {
            int aa = testAuthorIndices.get(ii);
            testFeatures[ii] = new SparseVector(V);
            ArrayList<Integer> docIndices = authorDocIndices.get(aa);
            if (docIndices != null) {
                for (int dd : authorDocIndices.get(aa)) {
                    for (int nn = 0; nn < words[dd].length; nn++) {
                        testFeatures[ii].change(words[dd][nn], 1.0);
                    }
                }
                for (int vv : testFeatures[ii].getIndices()) {
                    double tf = testFeatures[ii].get(vv);
                    double tfidf = Math.log(tf + 1) * debateVoteData.getIDFs()[vv];
                    testFeatures[ii].set(vv, tfidf);
                }
                if (isNormalizing()) {
                    testFeatures[ii].normalize();
                }
            }
        }
        outputFeatures(new File(tfidfFolder, DATAFILE + TEST_POSFIX), testFeatures);
        return modelName;
    }

    /**
     * Classify using TF.
     */
    protected String getTFFeatures(File foldResultFolder) throws Exception {
        String modelName = "tf";
        if (isNormalizing()) {
            modelName += "_normalized";
        }
        File tfFolder = new File(foldResultFolder, modelName);
        IOUtils.createFolder(tfFolder);

        int V = debateVoteData.getWordVocab().size();
        int[][] words = debateVoteData.getWords();

        int trA = trAuthorIds.length;
        SparseVector[] trainFeatures = new SparseVector[trA];
        for (int ii = 0; ii < trA; ii++) {
            int aa = trainAuthorIndices.get(ii);
            trainFeatures[ii] = new SparseVector(V);
            ArrayList<Integer> docIndices = authorDocIndices.get(aa);
            if (docIndices != null) {
                for (int dd : authorDocIndices.get(aa)) {
                    for (int nn = 0; nn < words[dd].length; nn++) {
                        trainFeatures[ii].change(words[dd][nn], 1.0);
                    }
                }
                if (isNormalizing()) {
                    trainFeatures[ii].normalize();
                }
            }
        }
        outputFeatures(new File(tfFolder, DATAFILE + TRAIN_POSFIX), trainFeatures);

        int teA = teAuthorIds.length;
        SparseVector[] testFeatures = new SparseVector[teA];
        for (int ii = 0; ii < teA; ii++) {
            int aa = testAuthorIndices.get(ii);
            testFeatures[ii] = new SparseVector(V);
            ArrayList<Integer> docIndices = authorDocIndices.get(aa);
            if (docIndices != null) {
                for (int dd : authorDocIndices.get(aa)) {
                    for (int nn = 0; nn < words[dd].length; nn++) {
                        testFeatures[ii].change(words[dd][nn], 1.0);
                    }
                }
                if (isNormalizing()) {
                    testFeatures[ii].normalize();
                }
            }
        }
        outputFeatures(new File(tfFolder, DATAFILE + TEST_POSFIX), testFeatures);
        return modelName;
    }

    private void runSVMRank(File foldResultFolder, String modelName,
            SparseVector[] trainingFeatures, SparseVector[] testFeatures) {
        if (trainingFeatures.length != trAuthorIds.length) {
            throw new MismatchRuntimeException(trainingFeatures.length, trAuthorIds.length);
        }
        if (testFeatures.length != teAuthorIds.length) {
            throw new MismatchRuntimeException(testFeatures.length, teAuthorIds.length);
        }
        String[] opts = new String[1];
        opts[0] = "-z p";

        SVMLight svm = new SVMLight();
        File svmFolder = new File(foldResultFolder, modelName + "_svm");
        IOUtils.createFolder(svmFolder);

        if (isTraining()) {
            File trainFile = new File(svmFolder, "data" + TRAIN_POSFIX);
            SVMUtils.outputSVMLightRankingFormat(trainFile, trainingFeatures, trAuthorLabels);
            svm.learn(opts, trainFile, new File(svmFolder, MODEL_FILE));
        }

        if (isTesting()) {
            File testFile = new File(svmFolder, "data" + TEST_POSFIX);
            SVMUtils.outputSVMLightRankingFormat(testFile, testFeatures, teAuthorLabels);

            File resultFile = new File(svmFolder, RESULT_FILE);
            svm.classify(null, testFile, new File(svmFolder, MODEL_FILE), resultFile);

            double[] predictions = svm.getPredictedValues(resultFile);
            File teResultFolder = new File(svmFolder, TEST_PREFIX + RESULT_FOLDER);
            IOUtils.createFolder(teResultFolder);
            evaluateRanking(teResultFolder, teAuthorIds, teAuthorLabels, predictions);
        }
    }

    /**
     * Run SVMLight.
     *
     * @param foldResultFolder
     * @param modelName
     * @param trainingFeatures
     * @param testFeatures
     */
    private void runSVMLightClassification(File foldResultFolder, String modelName,
            SparseVector[] trainingFeatures, SparseVector[] testFeatures) {
        if (trainingFeatures.length != trAuthorIds.length) {
            throw new MismatchRuntimeException(trainingFeatures.length, trAuthorIds.length);
        }
        if (testFeatures.length != teAuthorIds.length) {
            throw new MismatchRuntimeException(testFeatures.length, teAuthorIds.length);
        }

        SVMLight svm = new SVMLight();

        String[] opts = new String[2];
        opts[0] = "-z c";
        double j;
        if (cmd.hasOption("j")) { // from input
            j = CLIUtils.getDoubleArgument(cmd, "j", 1.0);
            modelName += "_svm_j-" + MiscUtils.formatDouble(j);
        } else { // decide from the trainining data
            int numNeg = 0;
            int numPos = 0;
            for (int ii = 0; ii < trAuthorLabels.length; ii++) {
                if (trAuthorLabels[ii] == 0) {
                    numNeg++;
                } else {
                    numPos++;
                }
            }
            j = (double) numNeg / numPos;
        }
        opts[1] = "-j " + j;

        File svmFolder = new File(foldResultFolder, modelName);
        IOUtils.createFolder(svmFolder);

        if (isTraining()) {
            File trainFile = new File(svmFolder, "data" + TRAIN_POSFIX);
            int[] svmLabels = new int[trAuthorLabels.length];
            for (int ii = 0; ii < svmLabels.length; ii++) {
                if (trAuthorLabels[ii] == 0) {
                    svmLabels[ii] = -1;
                } else {
                    svmLabels[ii] = 1;
                }
            }
            SVMUtils.outputSVMLightFormat(trainFile, trainingFeatures, svmLabels);
            svm.learn(opts, trainFile, new File(svmFolder, MODEL_FILE));
        }

        if (isTesting()) {
            File testFile = new File(svmFolder, "data" + TEST_POSFIX);
            int[] svmLabels = new int[teAuthorLabels.length];
            for (int ii = 0; ii < svmLabels.length; ii++) {
                if (teAuthorLabels[ii] == 0) {
                    svmLabels[ii] = -1;
                } else {
                    svmLabels[ii] = 1;
                }
            }
            SVMUtils.outputSVMLightFormat(testFile, testFeatures, svmLabels);

            File resultFile = new File(svmFolder, RESULT_FILE);
            svm.classify(null, testFile, new File(svmFolder, MODEL_FILE), resultFile);

            double[] predictions = svm.getPredictedValues(resultFile);
            File teResultFolder = new File(svmFolder, TEST_PREFIX + RESULT_FOLDER);
            IOUtils.createFolder(teResultFolder);
            evaluateClassification(teResultFolder, teAuthorIds, teAuthorLabels, predictions);
        }
    }

    /**
     * Run logistic regression using OWL-QN.
     *
     * @param foldResultFolder
     * @param modelName
     * @param trainingFeatures
     * @param testFeatures
     */
    private void runLogisticRegressionOWLQN(File foldResultFolder, String modelName,
            SparseVector[] trainingFeatures, SparseVector[] testFeatures,
            ArrayList<String> featureNames) {
        if (trainingFeatures.length != trAuthorIds.length) {
            throw new MismatchRuntimeException(trainingFeatures.length, trAuthorIds.length);
        }
        if (testFeatures.length != teAuthorIds.length) {
            throw new MismatchRuntimeException(testFeatures.length, teAuthorIds.length);
        }
        int K = trainingFeatures[0].getDimension();

        double l1 = CLIUtils.getDoubleArgument(cmd, "l1", 0.0);
        double l2 = CLIUtils.getDoubleArgument(cmd, "l2", 0.01);
        OWLQNLogisticRegression logreg = new OWLQNLogisticRegression(
                modelName + "_logreg-owqln", l1, l2, max_iters);
        File logregFolder = new File(foldResultFolder, logreg.getName());
        IOUtils.createFolder(logregFolder);

        if (cmd.hasOption("train")) {
            logreg.train(trainingFeatures, trAuthorLabels, K);
            logreg.output(new File(logregFolder, MODEL_FILE));

            double[] predictions = logreg.test(trainingFeatures);
            File trResultFolder = new File(logregFolder, TRAIN_PREFIX + RESULT_FOLDER);
            IOUtils.createFolder(trResultFolder);
            evaluateClassification(trResultFolder, trAuthorIds, trAuthorLabels, predictions);

            if (featureNames != null) {
                logreg.outputRankedWeights(new File(logregFolder, "weights.txt"), featureNames);
            }
        }

        if (cmd.hasOption("test")) {
            logreg.input(new File(logregFolder, MODEL_FILE));
            double[] predictions = logreg.test(testFeatures);

            File teResultFolder = new File(logregFolder, TEST_PREFIX + RESULT_FOLDER);
            IOUtils.createFolder(teResultFolder);
            evaluateClassification(teResultFolder, teAuthorIds, teAuthorLabels, predictions);
        }
    }

    /**
     * TODO: Implement this
     */
    private void runRidgeLogisticRegression(File foldResultFolder, String modelName,
            SparseVector[] trainingFeatures, SparseVector[] testFeatures,
            ArrayList<String> featureNames) {
        if (trainingFeatures.length != trAuthorIds.length) {
            throw new MismatchRuntimeException(trainingFeatures.length, trAuthorIds.length);
        }
        if (testFeatures.length != teAuthorIds.length) {
            throw new MismatchRuntimeException(testFeatures.length, teAuthorIds.length);
        }
        int K = trainingFeatures[0].getDimension();

        double sigma = CLIUtils.getDoubleArgument(cmd, "sigma", 2.5);
        double[] weights = new double[K];
        for (int kk = 0; kk < K; kk++) {
            weights[kk] = SamplerUtils.getGaussian(0.0, sigma);
        }
        RidgeLogisticRegressionOptimizable optimizable = new RidgeLogisticRegressionOptimizable(
                trAuthorLabels, weights, trainingFeatures, 0.0, sigma);
        File logregFolder = new File(foldResultFolder,
                modelName + "_logreg-lbfgs_s-" + MiscUtils.formatDouble(sigma));
        IOUtils.createFolder(logregFolder);

        LimitedMemoryBFGS optimizer = new LimitedMemoryBFGS(optimizable);

        if (cmd.hasOption("train")) {
            boolean converged = false;
            try {
                converged = optimizer.optimize();
            } catch (Exception ex) {
                ex.printStackTrace();
            }

//            logreg.train(trainingFeatures, trAuthorLabels, K);
//            logreg.output(new File(logregFolder, MODEL_FILE));
//            double[] predictions = logreg.test(trainingFeatures);
//            File trResultFolder = new File(logregFolder, TRAIN_PREFIX + RESULT_FOLDER);
//            IOUtils.createFolder(trResultFolder);
//            evaluateClassification(trResultFolder, trAuthorIds, trAuthorLabels, predictions);
//
//            if (featureNames != null) {
//                logreg.outputRankedWeights(new File(logregFolder, "weights.txt"), featureNames);
//            }
        }
    }

    /**
     * Run binary classifier using Liblinear. (Not working, need to fix)
     *
     * @param foldResultFolder
     * @param modelName
     * @param trainingFeatures
     * @param testFeatures
     */
    private void runLibLinear(File foldResultFolder, String modelName,
            SparseVector[] trainingFeatures, SparseVector[] testFeatures) throws Exception {
        if (trainingFeatures.length != trAuthorIds.length) {
            throw new MismatchRuntimeException(trainingFeatures.length, trAuthorIds.length);
        }
        if (testFeatures.length != teAuthorIds.length) {
            throw new MismatchRuntimeException(testFeatures.length, teAuthorIds.length);
        }

        double c = CLIUtils.getDoubleArgument(cmd, "c", 1);
        double p = CLIUtils.getDoubleArgument(cmd, "p", 0.1);
        double epsilon = CLIUtils.getDoubleArgument(cmd, "epsilon", 0.1);
        SolverType solveType = SolverType.L2R_LR;

        File logregFolder = new File(foldResultFolder,
                modelName + "_liblinear_st-" + solveType
                + "_c-" + MiscUtils.formatDouble(c)
                + "_e-" + MiscUtils.formatDouble(epsilon, 5));
        IOUtils.createFolder(logregFolder);

        if (cmd.hasOption("train")) {
            int A = trainingFeatures.length;
            int K = trainingFeatures[0].getDimension();

            Problem problem = new Problem();
            problem.l = A; // number of training examples
            problem.n = K;
            problem.x = new FeatureNode[A][K];
            for (int a = 0; a < A; a++) {
                for (int v = 0; v < K; v++) {
                    problem.x[a][v] = new FeatureNode(v + 1, trainingFeatures[a].get(v));
                }
            }

            for (Feature[] nodes : problem.x) {
                int indexBefore = 0;
                for (Feature n : nodes) {
                    if (n.getIndex() <= indexBefore) {
                        throw new IllegalArgumentException("Hello: feature nodes "
                                + "must be sorted by index in ascending order. "
                                + indexBefore + " vs. " + n.getIndex()
                                + "\t" + nodes.length);
                    }
                    indexBefore = n.getIndex();
                }
            }
            double[] dLabels = new double[A];
            for (int ii = 0; ii < A; ii++) {
                dLabels[ii] = trAuthorLabels[ii];
            }
            problem.y = dLabels;
            Parameter parameter = new Parameter(SolverType.L2R_LR_DUAL, c, epsilon, p);
            Model model = Linear.train(problem, parameter);
            model.save(new File(logregFolder, MODEL_FILE));

            double[] predictions = new double[A];
            for (int ii = 0; ii < A; ii++) {
                Feature[] instance = new Feature[K];
                for (int v = 0; v < K; v++) {
                    instance[v] = new FeatureNode(v + 1, trainingFeatures[ii].get(v));
                }
                double[] predProbs = new double[2];
                Linear.predictProbability(model, instance, predProbs);
                predictions[ii] = predProbs[1];
            }
            File trResultFolder = new File(logregFolder, TRAIN_PREFIX + RESULT_FOLDER);
            IOUtils.createFolder(trResultFolder);
            evaluateClassification(trResultFolder, trAuthorIds, trAuthorLabels, predictions);
        }

        if (cmd.hasOption("test")) {
            int A = testFeatures.length;
            int K = testFeatures[0].getDimension();

            Model model = Model.load(new File(logregFolder, MODEL_FILE));

            double[] predictions = new double[A];
            for (int ii = 0; ii < A; ii++) {
                Feature[] instance = new Feature[K];
                for (int v = 0; v < K; v++) {
                    instance[v] = new FeatureNode(v + 1, testFeatures[ii].get(v));
                }
                double[] predProbs = new double[2];
                Linear.predictProbability(model, instance, predProbs);
                predictions[ii] = predProbs[1];
            }
            File teResultFolder = new File(logregFolder, TEST_PREFIX + RESULT_FOLDER);
            IOUtils.createFolder(teResultFolder);
            evaluateClassification(teResultFolder, teAuthorIds, teAuthorLabels, predictions);
        }
    }

    @Override
    protected void runRandom(File foldResultFolder) {
        Random rand = new Random(1);
        double[] predictions = new double[teAuthorIds.length];
        for (int ii = 0; ii < predictions.length; ii++) {
            predictions[ii] = rand.nextDouble();
        }

        File teResultFolder = new File(new File(foldResultFolder, "random"),
                TEST_PREFIX + RESULT_FOLDER);
        IOUtils.createFolder(teResultFolder);
        if (affType == AffiliationType.ALL) {
            evaluateRanking(teResultFolder, teAuthorIds, teAuthorLabels, predictions);
        } else {
            evaluateClassification(teResultFolder, teAuthorIds, teAuthorLabels, predictions);
        }
    }

    /**
     * Create synthetic data to test classifiers.
     */
    protected void runTest(File foldResultFolder) throws Exception {
        int V = 100;
        int trA = trAuthorIds.length;
        SparseVector[] trainFeatures = new SparseVector[trA];
        for (int ii = 0; ii < trA; ii++) {
            trainFeatures[ii] = new SparseVector(V);
            int label = trAuthorLabels[ii];
            if (label == 1) {
                for (int vv = 0; vv < V; vv++) {
                    if (vv < V / 2) {
                        trainFeatures[ii].set(vv, SamplerUtils.getGaussian(5.0, 5.0));
                    } else {
                        trainFeatures[ii].set(vv, SamplerUtils.getGaussian(-5.0, 5.0));
                    }
                }
            }
        }

        int teA = teAuthorIds.length;
        SparseVector[] testFeatures = new SparseVector[teA];
        for (int ii = 0; ii < teA; ii++) {
            testFeatures[ii] = new SparseVector(V);
            int label = teAuthorLabels[ii];
            if (label == 1) {
                for (int vv = 0; vv < V; vv++) {
                    if (vv < V / 2) {
                        testFeatures[ii].set(vv, SamplerUtils.getGaussian(5.0, 5.0));
                    } else {
                        testFeatures[ii].set(vv, SamplerUtils.getGaussian(-5.0, 5.0));
                    }
                }
            }
        }

        String modelName = "test";
        ArrayList<String> featureNames = new ArrayList<>();
        for (int vv = 0; vv < V; vv++) {
            featureNames.add("Feature-" + vv);
        }
        runClassifier(foldResultFolder, modelName, trainFeatures, testFeatures, featureNames);
    }

    private static void evaluateClassification(File resultFolder,
            String[] authorIds, int[] authorLabels, double[] predictions) {
        PredictionUtils.outputClassificationPredictions(
                new File(resultFolder, PREDICTION_FILE),
                authorIds, authorLabels, predictions);
        PredictionUtils.outputClassificationResults(
                new File(resultFolder, RESULT_FILE),
                authorIds, authorLabels, predictions);
    }

    private static void evaluateRanking(File resultFolder,
            String[] authorIds, int[] authorLabels, double[] predictions) {
        // predictions
        RankingItemList<String> preds = new RankingItemList<String>();
        for (int ii = 0; ii < authorIds.length; ii++) {
            preds.addRankingItem(new RankingItem<String>(authorIds[ii], predictions[ii]));
        }
        preds.sortDescending();

        // groundtruth
        RankingItemList<String> truths = new RankingItemList<String>();
        for (int ii = 0; ii < authorIds.length; ii++) {
            truths.addRankingItem(new RankingItem<String>(authorIds[ii], authorLabels[ii]));
        }
        truths.sortDescending();

        RankingPerformance<String> rankPerf = new RankingPerformance<String>(preds,
                resultFolder.getAbsolutePath());
        rankPerf.computeAndOutputNDCGsNormalize(truths);
        double[] ndcgs = rankPerf.getNDCGs();
        ArrayList<Measurement> measurements = new ArrayList<>();
        measurements.add(new Measurement("NDCG@1", ndcgs[0]));
        measurements.add(new Measurement("NDCG@3", ndcgs[2]));
        measurements.add(new Measurement("NDCG@5", ndcgs[4]));
        measurements.add(new Measurement("NDCG@10", ndcgs[9]));
        measurements.add(new Measurement("NDCG", ndcgs[ndcgs.length - 1]));

        try {
            BufferedWriter writer = IOUtils.getBufferedWriter(new File(resultFolder, RESULT_FILE));
            for (Measurement m : measurements) {
                writer.write(m.getName() + "\t" + m.getValue() + "\n");
            }
            writer.close();
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException();
        }
    }

    private void outputFeatures(File outputFile, SparseVector[] features) {
        if (verbose) {
            logln("Outputing features to " + outputFile);
        }
        try {
            BufferedWriter writer = IOUtils.getBufferedWriter(outputFile);
            writer.write(features.length + "\n");
            for (SparseVector feature : features) {
                writer.write(SparseVector.output(feature) + "\n");
            }
            writer.close();
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while outputing to " + outputFile);
        }
    }

    private SparseVector[] inputFeatures(File inputFile) throws Exception {
        SparseVector[] features = null;
        try {
            BufferedReader reader = IOUtils.getBufferedReader(inputFile);
            int num = Integer.parseInt(reader.readLine());
            features = new SparseVector[num];
            for (int ii = 0; ii < num; ii++) {
                String line = reader.readLine();
                features[ii] = SparseVector.input(line);
            }
            reader.close();
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while loading from " + inputFile);
        }
        return features;
    }

    @Override
    public void preprocess() {
        if (verbose) {
            logln("Preprocessing ...");
        }
        try {
            loadFormattedData();
            getGOPs();
            outputCrossValidationData();
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while preprocessing");
        }
    }

    private void getGOPs() {
        gopAuthorList = new ArrayList<>();
        gopTPAffList = new ArrayList<>();
        for (int aa = 0; aa < debateVoteData.getAuthorVocab().size(); aa++) {
            String authorId = debateVoteData.getAuthorVocab().get(aa);
            String party = debateVoteData.getAuthorProperty(authorId, "party");
            if (party.equals("Republican")) {
                gopAuthorList.add(new Instance<Integer>(aa));
                gopTPAffList.add(getTeaPartyAffiliation(authorId));
            }
        }

        if (verbose) {
            logln("--- --- Total # authors: " + debateVoteData.getAuthorVocab().size());
            logln("--- --- Total # gops: " + gopAuthorList.size() + ", " + gopTPAffList.size());
            int numTps = 0;
            for (Integer tpAff : gopTPAffList) {
                if (tpAff == 1) {
                    numTps++;
                }
            }
            logln("--- --- Total # tps: " + numTps);
        }
    }

    private int getTeaPartyAffiliation(String authorId) {
        String icpsrId = debateVoteData.getAuthorProperty(authorId, GTLegislator.ICPSRID);
        if (this.affType == AffiliationType.TPCAUCUS) {
            return this.teapartyCaucusMapping.get(icpsrId);
        } else if (this.affType == AffiliationType.FREEDOMWORKS) {
            return this.fwEndorsementMapping.get(icpsrId);
        } else if (this.affType == AffiliationType.SARAHPALIN) {
            return this.spEndorsementMapping.get(icpsrId);
        } else if (this.affType == AffiliationType.TPEXPRESS) {
            return this.tpExpressMapping.get(icpsrId);
        } else if (this.affType == AffiliationType.ALL) {
            int tpcaucus = teapartyCaucusMapping.get(icpsrId);
            int tpexpress = tpExpressMapping.get(icpsrId);
            int fw = fwEndorsementMapping.get(icpsrId);
            int sp = spEndorsementMapping.get(icpsrId);
            return tpcaucus + tpexpress + fw + sp;
        } else if (this.affType == AffiliationType.ANY) {
            int tpcaucus = teapartyCaucusMapping.get(icpsrId);
            int tpexpress = tpExpressMapping.get(icpsrId);
            int fw = fwEndorsementMapping.get(icpsrId);
            int sp = spEndorsementMapping.get(icpsrId);
            int total = tpcaucus + tpexpress + fw + sp;
            return total > 0 ? 1 : 0;
        } else {
            throw new RuntimeException("Affiliation type " + affType + " not supported");
        }
    }

    private void outputCrossValidationData() throws Exception {
        if (verbose) {
            logln("--- Outputing cross validation data ... " + processedDataFolder);
        }
        CrossValidation<Integer, Instance<Integer>> cv = new CrossValidation<Integer, Instance<Integer>>(
                processedDataFolder,
                getCVFolder(),
                gopAuthorList);
        cv.stratify(gopTPAffList, numFolds, trToDevRatio);
        for (Fold fold : cv.getFolds()) {
            File foldFolder = new File(cv.getFolderPath(), fold.getFoldName());
            IOUtils.createFolder(foldFolder);
            outputFold(fold, foldFolder);
        }
    }

    private void outputFold(Fold<Integer, Instance<Integer>> fold, File foldFolder) throws Exception {
        BufferedWriter writer = IOUtils.getBufferedWriter(new File(foldFolder,
                fold.getFoldName() + Fold.TrainingExt));
        writer.write(fold.getTrainingInstances().size() + "\n");
        for (int trInst : fold.getTrainingInstances()) {
            int aa = fold.getInstanceList().get(trInst).getId();
            String authorId = debateVoteData.getAuthorVocab().get(aa);
            writer.write(trInst
                    + "\t" + aa
                    + "\t" + debateVoteData.getAuthorProperty(authorId, GTLegislator.ICPSRID)
                    + "\t" + debateVoteData.getAuthorProperty(authorId, GTLegislator.FW_ID)
                    + "\t" + debateVoteData.getAuthorProperty(authorId, GTLegislator.NAME)
                    + "\t" + gopTPAffList.get(trInst)
                    + "\n");
        }
        writer.close();

        writer = IOUtils.getBufferedWriter(new File(foldFolder,
                fold.getFoldName() + Fold.TestExt));
        writer.write(fold.getTestingInstances().size() + "\n");
        for (int teInst : fold.getTestingInstances()) {
            int aa = fold.getInstanceList().get(teInst).getId();
            String authorId = debateVoteData.getAuthorVocab().get(aa);
            writer.write(teInst
                    + "\t" + aa
                    + "\t" + debateVoteData.getAuthorProperty(authorId, GTLegislator.ICPSRID)
                    + "\t" + debateVoteData.getAuthorProperty(authorId, GTLegislator.FW_ID)
                    + "\t" + debateVoteData.getAuthorProperty(authorId, GTLegislator.NAME)
                    + "\t" + gopTPAffList.get(teInst)
                    + "\n");
        }
        writer.close();
    }

    private void inputFold(Fold<Integer, Instance<Integer>> fold) throws Exception {
        if (verbose) {
            logln("--- --- Inputing fold " + fold.getIndex() + " from " + fold.getFoldFolderPath());
        }

        String line;
        String[] sline;
        // training
        BufferedReader reader = IOUtils.getBufferedReader(new File(fold.getFoldFolderPath(),
                fold.getFoldName() + Fold.TrainingExt));
        int numTrains = Integer.parseInt(reader.readLine());
        this.trainAuthorIndices = new ArrayList<>();
        this.trAuthorIds = new String[numTrains];
        this.trAuthorLabels = new int[numTrains];
        int count = 0;
        while ((line = reader.readLine()) != null) {
            sline = line.split("\t");
            int instanceIdx = Integer.parseInt(sline[0]);
            int authorIdx = gopAuthorList.get(instanceIdx).getId();

            if (authorIdx != Integer.parseInt(sline[1])) {
                throw new MismatchRuntimeException(authorIdx, Integer.parseInt(sline[1]));
            }

            this.trainAuthorIndices.add(authorIdx);
            this.trAuthorIds[count] = debateVoteData.getAuthorVocab().get(authorIdx);
            this.trAuthorLabels[count] = this.gopTPAffList.get(instanceIdx);
            count++;
        }
        reader.close();

        trainVotes = new boolean[votes.length][];
        for (int aa = 0; aa < votes.length; aa++) {
            trainVotes[aa] = new boolean[votes[aa].length];
            Arrays.fill(trainVotes[aa], false);
            if (trainAuthorIndices.contains(aa)) {
                for (int bb = 0; bb < votes[aa].length; bb++) {
                    if (votes[aa][bb] != Vote.MISSING) {
                        trainVotes[aa][bb] = true;
                    }
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

        // test
        reader = IOUtils.getBufferedReader(new File(fold.getFoldFolderPath(),
                fold.getFoldName() + Fold.TestExt));
        int numTests = Integer.parseInt(reader.readLine());
        this.testAuthorIndices = new ArrayList<>();
        this.teAuthorIds = new String[numTests];
        this.teAuthorLabels = new int[numTests];
        count = 0;
        while ((line = reader.readLine()) != null) {
            sline = line.split("\t");
            int instanceIdx = Integer.parseInt(sline[0]);
            int authorIdx = gopAuthorList.get(instanceIdx).getId();

            if (authorIdx != Integer.parseInt(sline[1])) {
                throw new MismatchRuntimeException(authorIdx, Integer.parseInt(sline[1]));
            }

            this.testAuthorIndices.add(authorIdx);
            this.teAuthorIds[count] = debateVoteData.getAuthorVocab().get(authorIdx);
            this.teAuthorLabels[count] = this.gopTPAffList.get(instanceIdx);
            count++;
        }
        reader.close();

        testVotes = new boolean[votes.length][];
        for (int aa = 0; aa < votes.length; aa++) {
            testVotes[aa] = new boolean[votes[aa].length];
            Arrays.fill(testVotes[aa], false);
            if (testAuthorIndices.contains(aa)) {
                for (int bb = 0; bb < votes[aa].length; bb++) {
                    if (votes[aa][bb] != Vote.MISSING) {
                        testVotes[aa][bb] = true;
                    }
                }
            }
        }

        testDebateIndices = new ArrayList<>();
        for (int dd = 0; dd < debateVoteData.getWords().length; dd++) {
            int author = debateVoteData.getAuthors()[dd];
            if (testAuthorIndices.contains(author)) {
                this.testDebateIndices.add(dd);
            }
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

    @Override
    protected void evaluate(
            String resultFolder,
            String modelFolder,
            int numFolds,
            String phase,
            String resultFile) throws Exception {
        ArrayList<String> modelNames = new ArrayList<String>();
        HashMap<String, ArrayList<Measurement>>[] results = new HashMap[numFolds];
        for (int f = 0; f < numFolds; f++) {
            Fold fold = new Fold(f, null);
            String foldName = fold.getFoldName();
            File foldFolder = new File(resultFolder, foldName);
            if (!foldFolder.exists()) {
                continue;
            }
            File foldModelFolder = foldFolder;
            if (modelFolder != null) {
                foldModelFolder = new File(foldFolder, modelFolder);
            }
            if (!foldModelFolder.exists()) {
                continue;
            }
            if (verbose) {
                logln("--- Reading results from " + foldModelFolder);
            }

            String[] modelFolders = foldModelFolder.list();
            ArrayList<String> modelFolderList = new ArrayList<String>();
            modelFolderList.addAll(Arrays.asList(modelFolders));
            Collections.sort(modelFolderList);

            results[f] = new HashMap<String, ArrayList<Measurement>>();

            File foldSummary = new File(foldModelFolder, phase + SUMMARY_FILE);
            BufferedWriter writer = IOUtils.getBufferedWriter(foldSummary);
            if (verbose) {
                logln("--- Summarizing fold " + f + ". Writing to " + foldSummary);
            }

            int count = 0;
            for (String mFolder : modelFolderList) {
                File teResultFolder = new File(new File(foldModelFolder, mFolder),
                        phase + RESULT_FOLDER);
                if (!teResultFolder.exists()) {
                    continue;
                }
                File teResultFile = new File(teResultFolder, resultFile);
                if (!teResultFile.exists()) {
                    continue;
                }

                // read measurements
                BufferedReader reader = IOUtils.getBufferedReader(teResultFile);
                String line;
                ArrayList<Measurement> measurements = new ArrayList<Measurement>();
                while ((line = reader.readLine()) != null) {
                    Measurement m = new Measurement(line.split("\t")[0],
                            Double.parseDouble(line.split("\t")[1]));
                    measurements.add(m);
                }
                reader.close();

                if (!modelNames.contains(mFolder)) {
                    modelNames.add(mFolder);
                }
                results[f].put(mFolder, measurements);

                if (count == 0) {
                    writer.write("Model");
                    for (Measurement m : measurements) {
                        writer.write("\t" + m.getName());
                    }
                    writer.write("\n");
                }

                writer.write(mFolder);
                for (Measurement m : measurements) {
                    writer.write("\t" + m.getValue());
                }
                writer.write("\n");

                count++;
            }
            writer.close();
        }
        Collections.sort(modelNames);

        // summarize across folds
        File mergeFile = new File(resultFolder, phase + "merged-" + SUMMARY_FILE);
        File metaSumFile = new File(resultFolder, phase + "meta-" + SUMMARY_FILE);
        if (verbose) {
            System.out.println("--- Meta summarizing " + metaSumFile);
            System.out.println("--- Merge summarizing " + mergeFile);
        }
        ArrayList<String> measureNames = null;

        BufferedWriter writer = IOUtils.getBufferedWriter(metaSumFile);
        BufferedWriter mergeWriter = IOUtils.getBufferedWriter(mergeFile);
        mergeWriter.write("Model\tMetric\tValue\tFold\n");
        for (int f = 0; f < results.length; f++) {
            if (results[f] == null) {
                continue;
            }
            writer.write("Fold " + f + "\n");
            for (String modelName : modelNames) {
                ArrayList<Measurement> modelFoldMeasurements = results[f].get(modelName);
                if (modelFoldMeasurements != null) {
                    writer.write(modelName);
                    for (Measurement m : modelFoldMeasurements) {
                        writer.write("\t" + m.getValue());
                        mergeWriter.write(modelName
                                + "\t" + m.getName()
                                + "\t" + m.getValue()
                                + "\t" + f
                                + "\n");
                    }

                    if (measureNames == null) {
                        measureNames = new ArrayList<String>();
                        for (Measurement m : modelFoldMeasurements) {
                            measureNames.add(m.getName());
                        }
                    }
                    writer.write("\n");
                }
            }
            writer.write("\n\n");
        }

        // average
        if (measureNames != null) {
            for (String measure : measureNames) {
                ArrayList<RankingItem<String>> rankModels = new ArrayList<>();
                HashMap<String, ArrayList<Double>> modelVals = new HashMap<>();
                for (String model : modelNames) {
                    ArrayList<Double> vals = new ArrayList<Double>();
                    for (HashMap<String, ArrayList<Measurement>> result : results) {
                        if (result == null) {
                            continue;
                        }
                        ArrayList<Measurement> modelFoldMeasurements = result.get(model);
                        if (modelFoldMeasurements != null) {
                            for (Measurement m : modelFoldMeasurements) {
                                if (m.getName().equals(measure)) {
                                    vals.add(m.getValue());
                                }
                            }
                        }
                    }
                    if (vals.size() < numFolds) {
                        continue;
                    }

                    double avg = StatUtils.mean(vals);
                    rankModels.add(new RankingItem<String>(model, avg));
                    modelVals.put(model, vals);

                }
                Collections.sort(rankModels);

                writer.write(measure + "\n");
                writer.write("Model\tNum-folds\tAverage\tStdv\n");
                for (RankingItem<String> rankModel : rankModels) {
                    String model = rankModel.getObject();
                    ArrayList<Double> vals = modelVals.get(model);
                    double std = StatUtils.standardDeviation(vals);
                    writer.write(model
                            + "\t" + vals.size()
                            + "\t" + rankModel.getPrimaryValue()
                            + "\t" + std
                            + "\n");
                }
                writer.write("\n\n");
            }
        }
        writer.close();
        mergeWriter.close();
    }

    public static void main(String[] args) {
        try {
            long sTime = System.currentTimeMillis();

            addOptions();

            options.addOption("sampletrain", false, "sample train");
            options.addOption("sampletest", false, "sample test");
            options.addOption("sampleall", false, "sample all");
            options.addOption("sampleallfeature", false, "sample all");
            options.addOption("normalize", false, "normalize");

            addOption("aff-type", "Affiliation type");
            addOption("classifier", "Classifier");
            addOption("c", "c for LIBLINEAR");
            addOption("p", "p for LIBLINEAR");

            addOption("j", "j for SVM Light");

            addOption("T", "T");
            addOption("features", "List of features to combine");

            cmd = parser.parse(options, args);
            if (cmd.hasOption("help")) {
                CLIUtils.printHelp(getHelpString(
                        GOPExpt.class.getName()), options);
                return;
            }

            verbose = cmd.hasOption("v");
            debug = cmd.hasOption("d");

            Congress.setVerbose(verbose);
            Congress.setDebug(debug);
            TextDataset.setDebug(debug);
            TextDataset.setVerbose(verbose);

            String affType = CLIUtils.getStringArgument(cmd, "aff-type", "tpcaucus");

            AffiliationPredExpt expt = new AffiliationPredExpt(affType);
            expt.setup();
            String runMode = CLIUtils.getStringArgument(cmd, "run-mode", "run");
            switch (runMode) {
                case "preprocess":
                    expt.preprocess();
                    break;
                case "run":
                    expt.run();
                    break;
                case "evaluate":
                    expt.evaluate();
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
        } catch (RuntimeException | ParseException e) {
            e.printStackTrace();
        }
    }
}
