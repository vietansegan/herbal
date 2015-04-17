package votepredictor.textidealpoint.hierarchy;

import cc.mallet.optimize.LimitedMemoryBFGS;
import cc.mallet.types.Dirichlet;
import data.Author;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.Set;
import java.util.SortedSet;
import java.util.TreeSet;
import optimization.RidgeLinearRegressionOptimizable;
import sampler.unsupervised.LDA;
import sampling.likelihood.CascadeDirMult.PathAssumption;
import sampling.likelihood.DirMult;
import sampling.util.SparseCount;
import util.HTMLUtils;
import util.IOUtils;
import util.MiscUtils;
import util.MismatchRuntimeException;
import util.RankingItem;
import util.SamplerUtils;
import util.SparseVector;
import util.StatUtils;
import util.evaluation.Measurement;
import util.govtrack.GTLegislator;
import votepredictor.AbstractVotePredictor;
import votepredictor.BayesianIdealPoint;
import votepredictor.textidealpoint.AbstractTextSingleIdealPoint;

/**
 *
 * @author vietan
 */
public class HierMultSHDP extends AbstractTextSingleIdealPoint {

    public static final int PROPOSE_INDEX = 0;
    public static final int ASSIGN_INDEX = 1;
    public static final int NEW_CHILD_INDEX = -1;
    protected double topicAlpha;
    protected double frameAlphaGlobal;
    protected double frameAlphaLocal;
    protected double topicBeta;
    protected double frameBeta;

    protected double gamma; // eta variance
    protected double lambda;
    protected double epsilon;

    protected int threshold; // threshold on token count

    // input
    protected int K;
    protected int J;
    protected double[][] priorTopics;
    protected int[][] billWords;
    protected PathAssumption pathAssumption;

    // latent variables
    private Topic[] topics;

    // --- documents
    protected DirMult[] docTopics;
    protected SparseCount[][] docFramesCounts;
    private Frame[][] t;

    // --- bills
    protected DirMult[] billTopics;
    protected int[][] billZs;
    protected SparseVector[] billThetas;
    protected double[][] billTopicPriors;

    // --- author
    protected double[][] us;
    protected SparseCount[] auTopicCounts;
    protected SparseCount[][] auFrameCounts;

    // internal
    private int numFrameAssignmentChange;
    private ArrayList<String> topicVocab;
    private double sqrtRho;
    private ArrayList<Integer> authorList;
    private ArrayList<Integer> billList;
    private ArrayList<Integer> docList;
    private ArrayList<Integer> topicList;
    private double uniform;
    private double rate;
    private int initMaxIter;
    private boolean initRandomUs = true;
    private int numTokensAccepted;

    public HierMultSHDP() {
        this.basename = "Hier-Mult-SHDP";
    }

    public HierMultSHDP(String bname) {
        this.basename = bname;
    }

    public SparseCount[] getAuthorTopicCounts() {
        return this.auTopicCounts;
    }

    public SparseCount[][] getAuthorFrameCounts() {
        return this.auFrameCounts;
    }

    public void setInitRandomUs(boolean flag) {
        this.initRandomUs = flag;
    }

    public void setTopicVocab(ArrayList<String> topicVoc) {
        this.topicVocab = topicVoc;
    }

    public ArrayList<String> getTopicVocab() {
        return this.topicVocab;
    }

    public void setBillWords(int[][] billWords) {
        this.billWords = billWords;
    }

    public void setBillTopicPriors(double[][] billTopicPriors) {
        this.billTopicPriors = billTopicPriors;
    }

    public double[][] getBillThetas() {
        double[][] thetas = new double[B][];
        for (int bb = 0; bb < B; bb++) {
            thetas[bb] = this.billThetas[bb].dense();
        }
        return thetas;
    }

    public void setInitMaxIter(int initMaxIter) {
        this.initMaxIter = initMaxIter;
    }

    private double getAnnealingRate(double init, int t, int T) {
        return init / (1 + (double) t / T);
    }

    public void configure(HierMultSHDP sampler) {
        this.configure(sampler.folder,
                sampler.V,
                sampler.K,
                sampler.J,
                sampler.topicAlpha,
                sampler.frameAlphaGlobal,
                sampler.frameAlphaLocal,
                sampler.topicBeta,
                sampler.frameBeta,
                sampler.rho,
                sampler.sigma,
                sampler.gamma,
                sampler.lambda,
                sampler.epsilon,
                sampler.threshold,
                sampler.pathAssumption,
                sampler.priorTopics,
                sampler.initState,
                sampler.paramOptimized,
                sampler.BURN_IN,
                sampler.MAX_ITER,
                sampler.LAG,
                sampler.REP_INTERVAL);
    }

    public void configure(
            String folder,
            int V, int K, int J,
            double topicAlpha, double frameAlphaGlobal, double frameAlphaLocal,
            double topicBeta, double frameBeta,
            double rho,
            double sigma,
            double gamma,
            double lambda,
            double epsilon,
            int threshold,
            PathAssumption pathAssumption,
            double[][] priorTopics,
            InitialState initState,
            boolean paramOpt,
            int burnin, int maxiter, int samplelag, int repInt) {
        if (verbose) {
            logln("Configuring ...");
        }

        this.folder = folder;
        this.K = K;
        this.J = J;
        this.V = V;
        this.uniform = 1.0 / V;

        this.hyperparams = new ArrayList<Double>();
        this.topicAlpha = topicAlpha;
        this.frameAlphaGlobal = frameAlphaGlobal;
        this.frameAlphaLocal = frameAlphaLocal;
        this.topicBeta = topicBeta;
        this.frameBeta = frameBeta;

        this.rho = rho;
        this.sqrtRho = Math.sqrt(rho);
        this.sigma = sigma;
        this.gamma = gamma;
        this.lambda = lambda;
        this.epsilon = epsilon;
        this.threshold = threshold;
        this.priorTopics = priorTopics;
        this.wordWeightType = WordWeightType.NONE;
        this.pathAssumption = pathAssumption;

        this.sampledParams = new ArrayList<ArrayList<Double>>();
        this.sampledParams.add(cloneHyperparameters());

        this.BURN_IN = burnin;
        this.MAX_ITER = maxiter;
        this.LAG = samplelag;
        this.REP_INTERVAL = repInt;

        this.initState = initState;
        this.paramOptimized = paramOpt;
        this.prefix += initState.toString();
        this.report = true;

        StringBuilder str = new StringBuilder();
        str.append(this.prefix)
                .append("_").append(basename)
                .append("_B-").append(BURN_IN)
                .append("_M-").append(MAX_ITER)
                .append("_L-").append(LAG)
                .append("_K-").append(this.K)
                .append(this.J == 0 ? "" : ("_J-" + this.J))
                .append("_ta-").append(formatter.format(this.topicAlpha))
                .append("_fag-").append(formatter.format(this.frameAlphaGlobal))
                .append("_fal-").append(formatter.format(this.frameAlphaLocal))
                .append("_tb-").append(formatter.format(this.topicBeta))
                .append("_fb-").append(formatter.format(this.frameBeta))
                .append("_r-").append(formatter.format(this.rho))
                .append("_s-").append(formatter.format(this.sigma))
                .append("_g-").append(formatter.format(this.gamma))
                .append("_l-").append(formatter.format(this.lambda))
                .append("_e-").append(MiscUtils.formatDouble(this.epsilon, 10))
                .append("_t-").append(threshold);
        str.append("_opt-").append(this.paramOptimized);
        str.append("_prior-").append(this.priorTopics != null);
        str.append(this.J > 0 ? "" : ("_path-" + this.pathAssumption));
        this.name = str.toString();

        if (verbose) {
            logln("--- folder\t" + folder);
            logln("--- num topics:\t" + this.K);
            logln("--- num frames:\t" + this.J);
            logln("--- num word types:\t" + this.V);
            logln("--- topic alpha:\t" + MiscUtils.formatDouble(this.topicAlpha));
            logln("--- frame alpha global:\t" + MiscUtils.formatDouble(this.frameAlphaGlobal));
            logln("--- frame alpha local:\t" + MiscUtils.formatDouble(this.frameAlphaLocal));
            logln("--- topic beta:\t" + MiscUtils.formatDouble(this.topicBeta));
            logln("--- frame beta:\t" + MiscUtils.formatDouble(this.frameBeta));
            logln("--- rho:\t" + MiscUtils.formatDouble(this.rho));
            logln("--- sqrt rho:\t" + MiscUtils.formatDouble(this.sqrtRho));
            logln("--- sigma:\t" + MiscUtils.formatDouble(this.sigma));
            logln("--- gamma:\t" + MiscUtils.formatDouble(this.gamma));
            logln("--- epsilon:\t" + MiscUtils.formatDouble(this.epsilon));
            logln("--- threshold:\t" + this.threshold);
            logln("--- burn-in:\t" + BURN_IN);
            logln("--- max iter:\t" + MAX_ITER);
            logln("--- sample lag:\t" + LAG);
            logln("--- paramopt:\t" + paramOptimized);
            logln("--- initialize:\t" + initState);
            logln("--- has prior? " + (priorTopics != null));
            logln("--- path assumption:\t" + this.pathAssumption);
        }
    }

    public double[][] getEstimatedUs() {
        double[][] estUs = new double[A][K];
        for (int aa = 0; aa < A; aa++) {
            for (int kk = 0; kk < K; kk++) {
                if (auTopicCounts[aa].getCount(kk) != 0) {
                    estUs[aa][kk] = getLexicalU(aa, kk);
                } else {
                    estUs[aa][kk] = -1;
                }
            }
        }
        return estUs;
    }

    public double[][] getMultiUs() {
        return this.us;
    }

    @Override
    public void initialize() {
        initialize(priorTopics);
    }

    public void initialize(double[][] seededTopics) {
        if (verbose) {
            logln("Initializing ...");
        }
        docList = new ArrayList<>();
        for (int dd = 0; dd < D; dd++) {
            docList.add(dd);
        }

        authorList = new ArrayList<>();
        for (int aa = 0; aa < A; aa++) {
            authorList.add(aa);
        }

        billList = new ArrayList<>();
        for (int bb = 0; bb < B; bb++) {
            billList.add(bb);
        }

        this.topicList = new ArrayList<>();
        for (int kk = 0; kk < K; kk++) {
            topicList.add(kk);
        }

        iter = INIT;
        isReporting = true;

        initializeBillThetas();
        initializeModelStructure(seededTopics);
        initializeDataStructure();
        initializeIdealPoint();
        initializeAssignments();

        if (debug) {
            validate("Initialized");
        }

        if (verbose) {
            logln("--- Done initializing. \n" + getCurrentState());
            getLogLikelihood();
        }
    }

    @Override
    protected void initializeIdealPoint() {
        BayesianIdealPoint bip = new BayesianIdealPoint();
        bip.configure(1.0, epsilon, 50000, 0.0, sigma);
        bip.setTrain(votes, authorIndices, billIndices, validVotes);

        File bipFolder = new File(folder, bip.getName());
        File bipFile = new File(bipFolder, "model");

        if (bipFile.exists()) {
            if (verbose) {
                logln("B.I.P. file exists. Loading from " + bipFile);
            }
            bip.input(bipFile);
        } else {
            if (verbose) {
                logln("B.I.P. file not found. Running and outputing to " + bipFile);
            }
            IOUtils.createFolder(bipFolder);
            bip.train();
            bip.output(bipFile);
        }
        this.u = bip.getUs();
        this.x = bip.getXs();
        this.y = bip.getYs(); // initialize y to make sure the signs are correct

        this.setInitRandomUs(false);

//        this.initializeMultiIdealPoints();
        this.us = new double[A][K];
        for (int aa = 0; aa < A; aa++) {
            for (int kk = 0; kk < K; kk++) {
                this.us[aa][kk] = this.u[aa];
            }
        }
    }

    private void initializeMultiIdealPoints() {
        double stdvU = StatUtils.standardDeviation(this.u);
        double stdvX = StatUtils.standardDeviation(this.x);
        if (verbose) {
            logln(">>> Stdv init u: " + stdvU);
            logln(">>> Stdv init x: " + stdvX);
            logln(">>> Stdv init y: " + StatUtils.standardDeviation(this.y));
        }

        this.us = new double[A][K];
        for (int aa = 0; aa < A; aa++) {
            for (int kk = 0; kk < K; kk++) {
                if (initRandomUs) {
                    this.us[aa][kk] = SamplerUtils.getGaussian(0.0, stdvU * stdvU);

                } else {
                    this.us[aa][kk] = this.u[aa];
                }
            }
        }

        this.x = new double[B];
        for (int bb = 0; bb < B; bb++) {
            this.x[bb] = SamplerUtils.getGaussian(0.0, stdvX * stdvX);
        }

        if (verbose) {
            double absU = 0.0;
            for (int aa = 0; aa < A; aa++) {
                for (int kk = 0; kk < K; kk++) {
                    absU += Math.abs(us[aa][kk]);
                }
            }
            double absX = 0.0;
            double absY = 0.0;
            for (int bb = 0; bb < B; bb++) {
                absX += Math.abs(x[bb]);
                absY += Math.abs(y[bb]);
            }
            logln(">>> AbsU: " + absU / (A * K));
            logln(">>> AbsX: " + absX / B);
            logln(">>> AbsY: " + absY / B);
        }

        int initLag = MiscUtils.getRoundStepSize(initMaxIter, 10);
        double thres = 1E-6;
        double curVal = 0.0;
        double initEpsilon = epsilon;
        double initSigma = sigma;

        File tipFolder = new File(folder, "topic-ideal-point"
                + "-m_" + initMaxIter
                + "-t_" + MiscUtils.formatDouble(thres, 10)
                + "-s_" + MiscUtils.formatDouble(initSigma)
                + "-e_" + MiscUtils.formatDouble(initEpsilon));
        IOUtils.createFolder(tipFolder);
        File initIPFile = new File(tipFolder, this.basename + "_initU-" + initRandomUs + ".zip");
        if (initIPFile.exists()) {
            if (verbose) {
                logln("Init ideal point file exists. Loading from " + initIPFile);
            }
            inputAuthorIdealPoints(initIPFile.getAbsolutePath());
            inputBillIdealPoints(initIPFile.getAbsolutePath());
        } else {
            if (verbose) {
                logln("Init ideal point file not found. " + initIPFile);
                logln("Running ... ");
            }
            for (int ii = 0; ii < initMaxIter; ii++) {
                rate = getAnnealingRate(initEpsilon, ii, MAX_ITER);
                initializeUs();
                initializeXY();

                if (verbose && ii % initLag == 0) {
                    logln("Init ii = " + ii + " / " + initMaxIter);
                    SparseVector[] predictions = predictInMatrix();
                    ArrayList<Measurement> measurements = AbstractVotePredictor
                            .evaluate(votes, validVotes, predictions);

                    double val = 0.0;
                    for (Measurement m : measurements) {
                        logln(">>> i >>> " + m.getName() + ": " + m.getValue());
                        if (m.getName().equals("avg-loglikelihood")) {
                            val = m.getValue();
                        }
                    }

                    double absU = 0.0;
                    for (int aa = 0; aa < A; aa++) {
                        for (int kk = 0; kk < K; kk++) {
                            absU += Math.abs(us[aa][kk]);
                        }
                    }
                    double absX = 0.0;
                    double absY = 0.0;
                    for (int bb = 0; bb < B; bb++) {
                        absX += Math.abs(x[bb]);
                        absY += Math.abs(y[bb]);
                    }
                    double diff = Math.abs(val - curVal);

                    logln(">>> AbsU: " + absU / (A * K));
                    logln(">>> AbsX: " + absX / B);
                    logln(">>> AbsY: " + absY / B);
                    logln(">>> Diff: " + diff);
                    logln(">>> Rate: " + rate);

                    System.out.println();

                    if (diff < thres) {
                        logln(">>> >>> Diff = " + diff + ". Terminating ...");
                        break;
                    } else {
                        curVal = val;
                    }
                }
            }
            // train author scores
            StringBuilder authorStr = new StringBuilder();
            authorStr.append(A).append("\n");
            for (int aa = 0; aa < A; aa++) {
                authorStr.append(MiscUtils.arrayToString(us[aa])).append("\n");
            }

            // train bill scores
            StringBuilder billStr = new StringBuilder();
            billStr.append(B).append("\n");
            for (int bb = 0; bb < B; bb++) {
                billStr.append(x[bb]).append("\n");
                billStr.append(y[bb]).append("\n");
            }

            ArrayList<String> contentStrs = new ArrayList<>();
            contentStrs.add(authorStr.toString());
            contentStrs.add(billStr.toString());

            String filename = IOUtils.removeExtension(IOUtils.getFilename(initIPFile.getAbsolutePath()));
            ArrayList<String> entryFiles = new ArrayList<>();
            entryFiles.add(filename + AuthorFileExt);
            entryFiles.add(filename + BillFileExt);
            try {
                if (verbose) {
                    logln("Outputing to " + initIPFile);
                }
                this.outputZipFile(initIPFile.getAbsolutePath(), contentStrs, entryFiles);
            } catch (Exception e) {
                e.printStackTrace();
                throw new RuntimeException("Exception while outputing to " + initIPFile);
            }
        }
    }

    protected void initializeUs() {
        Collections.shuffle(authorList);
        for (int aa : authorList) {
            if (!validAs[aa]) {
                continue;
            }
            Collections.shuffle(topicList);
            for (int kk : topicList) { // update 
                double llh = 0.0;
                for (int bb = 0; bb < B; bb++) {
                    if (isValidVote(aa, bb)) {
                        double dotprod = y[bb] + x[bb] * billThetas[bb].dotProduct(us[aa]);
                        double score = Math.exp(dotprod);
                        double prob = score / (1 + score);
                        llh += x[bb] * billThetas[bb].get(kk) * (getVote(aa, bb) - prob);
                    }
                }
                us[aa][kk] += (llh - us[aa][kk] / sigma) * rate / B;
            }
        }
    }

    public void initializeXY() {
        Collections.shuffle(billList);
        for (int bb : billList) {
            if (!validBs[bb]) {
                continue;
            }
            double llhX = 0.0;
            double llhY = 0.0;
            for (int aa = 0; aa < A; aa++) {
                if (isValidVote(aa, bb)) {
                    double dotprod = billThetas[bb].dotProduct(us[aa]);
                    double score = Math.exp(y[bb] + x[bb] * dotprod);
                    double prob = score / (1 + score);
                    llhX += (getVote(aa, bb) - prob) * dotprod;
                    llhY += getVote(aa, bb) - prob;
                }
            }
            x[bb] += (llhX - x[bb] / sigma) * rate / A;
            y[bb] += (llhY - y[bb] / sigma) * rate / A;
        }
    }

    protected void initializeBillThetas() {
        LDA lda = runLDA(billWords, K, V, billTopicPriors, priorTopics,
                0.5, 1000, 250, 500, 50);
        this.billTopics = lda.getDocTopics();
        this.billZs = lda.getZs();

        // empirical distributions over topics
        this.billThetas = new SparseVector[B];
        for (int bb = 0; bb < B; bb++) {
            this.billThetas[bb] = new SparseVector(this.billTopics[bb].getDistribution());
        }
    }

    protected void initializeModelStructure(double[][] seededTopics) {
        if (seededTopics != null && seededTopics.length != K) {
            throw new RuntimeException("Mismatch" + ". K = " + K
                    + ". # prior topics = " + seededTopics.length);
        }

        this.topics = new Topic[K];
        for (int kk = 0; kk < K; kk++) {
            DirMult topic;
            if (seededTopics != null) {
                topic = new DirMult(V, topicBeta * V, seededTopics[kk]);
            } else {
                topic = new DirMult(V, topicBeta * V, uniform);
            }
            this.topics[kk] = new Topic(kk, topic);

            if (this.J > 0) {
                this.topics[kk].psi = new SparseVector(J);
                for (int jj = 0; jj < J; jj++) {
                    DirMult framePhi = new DirMult(V, frameBeta * V, uniform);
                    double eta = SamplerUtils.getGaussian(0.0, gamma);
                    Frame frame = new Frame(jj, kk, iter, framePhi, eta);
                    frame.changeToNormal();
                    this.topics[kk].createNewComponent(jj, frame);
                    this.topics[kk].psi.set(jj, 1.0 / J);
                }
            }
        }
    }

    protected void initializeDataStructure() {
        docTopics = new DirMult[D];
        docFramesCounts = new SparseCount[D][K];
        for (int dd = 0; dd < D; dd++) {
            docTopics[dd] = new DirMult(K, topicAlpha * K, 1.0 / K);
            for (int kk = 0; kk < K; kk++) {
                docFramesCounts[dd][kk] = new SparseCount();
            }
        }

        t = new Frame[D][];
        for (int d = 0; d < D; d++) {
            t[d] = new Frame[words[d].length];
        }

        auTopicCounts = new SparseCount[A];
        auFrameCounts = new SparseCount[A][K];
        for (int aa = 0; aa < A; aa++) {
            auTopicCounts[aa] = new SparseCount();
            for (int kk = 0; kk < K; kk++) {
                auFrameCounts[aa][kk] = new SparseCount();
            }
        }
    }

    protected void initializeAssignments() {
        switch (initState) {
            case RANDOM:
                initializeRandomAssignments();
                break;
            default:
                throw new RuntimeException("Initialization " + initState + " not supported");
        }
    }

    protected void initializeRandomAssignments() {
        if (verbose) {
            logln("--- Initializing random assignments ...");
        }

        for (int ii = 0; ii < 10; ii++) {
            if (ii == 0) {
//                sampleZs_MH(!REMOVE, ADD, !REMOVE, ADD, !OBSERVED, EXTEND);
                sampleZs(!REMOVE, ADD, !REMOVE, ADD, !OBSERVED, EXTEND);
            } else {
//                sampleZs_MH(REMOVE, ADD, REMOVE, ADD, !OBSERVED, EXTEND);
                sampleZs(REMOVE, ADD, REMOVE, ADD, !OBSERVED, EXTEND);
            }

            if (this.J == 0) {
                updatePsis();
            }
        }
    }

    /**
     * Make prediction on held-out votes of known legislators and known votes.
     *
     * @return Predicted probabilities
     */
    @Override
    public SparseVector[] predictInMatrix() {
        SparseVector[] predictions = new SparseVector[validVotes.length];
        for (int aa = 0; aa < A; aa++) {
            int author = authorIndices.get(aa);
            predictions[author] = new SparseVector(validVotes[author].length);
            for (int bb = 0; bb < B; bb++) {
                int bill = billIndices.get(bb);
                if (isValidVote(aa, bb)) {
                    double dotprod = y[bb] + x[bb] * billThetas[bb].dotProduct(us[aa]);
                    double score = Math.exp(dotprod);
                    double prob = score / (1.0 + score);
                    predictions[author].set(bill, prob);
                }
            }
        }
        return predictions;
    }

    @Override
    public SparseVector[] predictOutMatrix() {
        SparseVector[] predictions = new SparseVector[validVotes.length];
        for (int aa = 0; aa < A; aa++) {
            int author = authorIndices.get(aa);
            predictions[author] = new SparseVector(validVotes[author].length);
            for (int bb = 0; bb < B; bb++) {
                int bill = billIndices.get(bb);
                if (isValidVote(aa, bb)) {
                    double dotprod = y[bb];
                    for (int kk = 0; kk < K; kk++) {
                        double ip;
                        if (isBackedOff(aa, kk)) {
                            ip = us[aa][kk];
                        } else {
                            ip = getLexicalU(aa, kk);
                        }
                        dotprod += x[bb] * billThetas[bb].get(kk) * ip;
                    }
                    double score = Math.exp(dotprod);
                    double prob = score / (1.0 + score);
                    predictions[author].set(bill, prob);
                }
            }
        }
        return predictions;
    }

    private boolean isBackedOff(int aa, int kk) {
        return auTopicCounts[aa].getCount(kk) < this.threshold;
    }

    /**
     * Get the estimation of the ideal point of voter aa on topic kk.
     *
     * @param aa Voter
     * @param kk Topic
     */
    protected double getLexicalU(int aa, int kk) {
        int authorTopicCount = auTopicCounts[aa].getCount(kk);
        if (isBackedOff(aa, kk)) {
            throw new RuntimeException("Author " + aa + " does not talk much about topic " + kk);
        }
        double lexMean = 0.0;
        for (int jj : auFrameCounts[aa][kk].getIndices()) {
            double frameFrac = (double) auFrameCounts[aa][kk].getCount(jj) / authorTopicCount;
            lexMean += topics[kk].getFrame(jj).eta * frameFrac;
        }
        return lexMean;
    }

    @Override
    public String getCurrentState() {
        String str = "\n\nIter " + iter + "/" + MAX_ITER
                + "\t @ " + Thread.currentThread().getId();
        if (iter < BURN_IN) {
            logln("--- Burning in. " + str);
        } else {
            logln("--- Sampling. " + str);
        }

        printGlobalTreeSummery();

        logln("--- Evaluating ...");
        SparseVector[] predictions = predictInMatrix();
        ArrayList<Measurement> measurements = AbstractVotePredictor.evaluate(votes, validVotes, predictions);
        for (Measurement m : measurements) {
            logln(">>> i >>> " + m.getName() + ": " + m.getValue());
        }

        predictions = predictOutMatrix();
        measurements = AbstractVotePredictor.evaluate(votes, validVotes, predictions);
        for (Measurement m : measurements) {
            logln(">>> o >>> " + m.getName() + ": " + m.getValue());
        }

        // author mse
        double mse = 0.0;
        int count = 0;
        for (int aa = 0; aa < A; aa++) {
            for (int kk = 0; kk < K; kk++) {
                if (!isBackedOff(aa, kk)) {
                    double diff = us[aa][kk] - getLexicalU(aa, kk);
                    mse += diff * diff;
                    count++;
                }
            }
        }
        logln("--- Regression MSE: " + (mse / (count)));
        logln("--- Non-zero Count: " + count + " / " + (A * K)
                + ". " + MiscUtils.formatDouble((double) count / (A * K)));

        double absU = 0.0;
        for (int aa = 0; aa < A; aa++) {
            for (int kk = 0; kk < K; kk++) {
                absU += Math.abs(us[aa][kk]);
            }
        }
        double absX = 0.0;
        double absY = 0.0;
        for (int bb = 0; bb < B; bb++) {
            absX += Math.abs(x[bb]);
            absY += Math.abs(y[bb]);
        }
        logln("--- AbsU = " + absU / (A * K));
        logln("--- AbsX = " + absX / B);
        logln("--- AbsY = " + absY / B);
        logln("--- rate = " + rate);

        return str;
    }

    @Override
    public void iterate() {
        updateEtas();

//        sampleZs_MH(REMOVE, ADD, REMOVE, ADD, OBSERVED, EXTEND);
        sampleZs(REMOVE, ADD, REMOVE, ADD, OBSERVED, EXTEND);

        if (this.J == 0) {
            updatePsis();
        }

//        updateUXY();
    }

    protected long sampleZs(
            boolean removeFromModel, boolean addToModel,
            boolean removeFromData, boolean addToData,
            boolean observe, boolean extend) {
        if (isReporting) {
            logln("+++ Sampling assignments ...");
        }
        long sTime = System.currentTimeMillis();
        numTokensChanged = 0;
        numFrameAssignmentChange = 0;
        numTokensAccepted = 0;

        for (int d = 0; d < D; d++) {
            int aa = authors[d];
            for (int n = 0; n < words[d].length; n++) {
                if (removeFromData) {
                    docTopics[d].decrement(t[d][n].topicIndex);
                    docFramesCounts[d][t[d][n].topicIndex].decrement(t[d][n].index);

                    auTopicCounts[aa].decrement(t[d][n].topicIndex);
                    auFrameCounts[aa][t[d][n].topicIndex].decrement(t[d][n].index);
                }

                if (removeFromModel) {
                    topics[t[d][n].topicIndex].phi.decrement(words[d][n]);
                    t[d][n].phi.decrement(words[d][n]);
                    if (t[d][n].phi.isEmpty()) {
                        topics[t[d][n].topicIndex].removeComponent(t[d][n].index);
                        topics[t[d][n].topicIndex].psi.remove(t[d][n].index);
                        t[d][n].changeToNew();

                        // debug
                        if (docFramesCounts[d][t[d][n].topicIndex].getCount(t[d][n].index) != 0) {
                            throw new RuntimeException();
                        }
                        if (auFrameCounts[aa][t[d][n].topicIndex].getCount(t[d][n].index) != 0) {
                            throw new RuntimeException();
                        }
                    }
                }

                Frame sampledFrame = sampleFrame(d, n, observe, extend);
                if (t[d][n] != null) {
                    if (t[d][n].topicIndex != sampledFrame.topicIndex) {
                        numTokensChanged++;
                    }
                    if (!t[d][n].equals(sampledFrame)) {
                        numFrameAssignmentChange++;
                    }
                }
                t[d][n] = sampledFrame;

                if (addToModel) {
                    topics[t[d][n].topicIndex].phi.increment(words[d][n]);
                    t[d][n].phi.increment(words[d][n]);
                }

                if (addToData) {
                    docTopics[d].increment(t[d][n].topicIndex);
                    docFramesCounts[d][t[d][n].topicIndex].increment(t[d][n].index);

                    auTopicCounts[aa].increment(t[d][n].topicIndex);
                    auFrameCounts[aa][t[d][n].topicIndex].increment(t[d][n].index);
                }

                // if a new frame is created, update psi and change the frame node status
                if (this.J == 0 && t[d][n].isNew) { // if accept
                    topics[t[d][n].topicIndex].createNewComponent(t[d][n].index, t[d][n]);
                    topics[t[d][n].topicIndex].psi.set(t[d][n].index,
                            frameAlphaGlobal / topics[t[d][n].topicIndex].phi.getCountSum()); // temp weight
                    topics[t[d][n].topicIndex].updatePsi();
                    t[d][n].changeToNormal();
                }
            }
        }

        long eTime = System.currentTimeMillis() - sTime;
        if (isReporting) {
            logln("--- --- time: " + eTime);
            logln("--- --- # tokens: " + numTokens
                    + ". # topic asgns: " + numTokensChanged
                    + " (" + MiscUtils.formatDouble((double) numTokensChanged / numTokens) + ")"
                    + ". # frame asgns: " + numFrameAssignmentChange
                    + " (" + MiscUtils.formatDouble((double) numFrameAssignmentChange / numTokens) + ")"
                    + ". # accepted: " + numTokensAccepted
                    + " (" + MiscUtils.formatDouble((double) numTokensAccepted / numTokens) + ")");

        }
        return eTime;
    }

    /**
     * Sample a frame node for a token in the proposal step of MH algorithm.
     *
     * @param dd Document index
     * @param nn Token index
     * @param observed
     * @param extend
     */
    private Frame sampleFrame(int dd, int nn, boolean observed, boolean extend) {
        int aa = authors[dd];

        // sample topic
        double[] probs = new double[K];
        for (int kk = 0; kk < K; kk++) {
            probs[kk] = docTopics[dd].getProbability(kk)
                    * topics[kk].getPhi(words[dd][nn]);
        }
        int kk = SamplerUtils.scaleSample(probs);
        Topic topic = topics[kk];

        // sample frame
        ArrayList<Frame> frameList = new ArrayList<>();
        ArrayList<Double> logprobList = new ArrayList<>();
        double norm = docFramesCounts[dd][kk].getCountSum() + frameAlphaLocal;
        double rawMean = 0.0;
        if (isBackedOff(aa, kk) && observed) {
            for (int jj : auFrameCounts[aa][kk].getIndices()) {
                rawMean += auFrameCounts[aa][kk].getCount(jj) * topic.getFrame(jj).eta;
            }
        }

        // --- for existing frame node
        for (Frame frame : topic.getFrames()) {
            frameList.add(frame);
            int jj = frame.index;
            double pathprob = (docFramesCounts[dd][kk].getCount(jj)
                    + frameAlphaLocal * topic.psi.get(jj)) / norm;
            double wordLlh = Math.log(pathprob * frame.getPhi(words[dd][nn]));

            double resLlh = 0.0;
            if (isBackedOff(aa, kk) && observed) {
                double lexMean = (rawMean + frame.eta) / (auTopicCounts[aa].getCount(kk) + 1.0);
                double mean = lexMean;
                resLlh = StatUtils.logNormalProbability(us[aa][kk], mean, sqrtRho);
            }
            double logprob = wordLlh + resLlh;
            logprobList.add(logprob);
        }

        // --- for new frame node
        double eta = 0.0;
        if (extend && this.J == 0) {
            frameList.add(null);
            eta = SamplerUtils.getGaussian(0.0, gamma);
            double pathprob = frameAlphaLocal * topic.psi.get(NEW_CHILD_INDEX) / norm;
            double logprob = Math.log(pathprob * uniform);
            if (isBackedOff(aa, kk) && observed) {
                double lexMean = (rawMean + eta) / (auTopicCounts[aa].getCount(kk) + 1.0);
                double mean = lexMean;
                double resLlh = StatUtils.logNormalProbability(us[aa][kk], mean, sqrtRho);
                logprob += resLlh;
            }
            logprobList.add(logprob);
        }

        int sampledIdx = SamplerUtils.logMaxRescaleSample(logprobList);

        if (sampledIdx == logprobList.size()) {
            for (int ii = 0; ii < logprobList.size(); ii++) {
                Frame frame = frameList.get(ii);

                double pathprob = (docFramesCounts[dd][kk].getCount(frame.index)
                        + frameAlphaLocal * topic.psi.get(frame.index)) / norm;
                double logprob = Math.log(pathprob * frame.getPhi(words[dd][nn]));
                double mean = (rawMean + frame.eta) / (auTopicCounts[aa].getCount(kk) + 1.0);
                double resllh = StatUtils.logNormalProbability(us[aa][kk], mean, sqrtRho);

                System.out.println("iter = " + iter
                        + ". dd = " + dd
                        + ". nn = " + nn);
                System.out.println("ii = " + ii
                        + ". " + frameList.get(ii).index
                        + ". " + frameList.get(ii).phi.getCountSum()
                        + ". path: " + pathprob + " (" + logprob + ")"
                        + ". res: " + resllh
                        + ". us: " + us[aa][kk]
                        + ". mean: " + mean
                        + ". sqrt: " + sqrtRho
                        + ". " + logprobList.get(ii));
            }
        }

        Frame sampledFrame = frameList.get(sampledIdx);
        if (sampledFrame == null) { // create new node
            assert this.J == 0;
            int frameIdx = topic.getNextIndex();
            DirMult framePhi = new DirMult(V, frameBeta * V, uniform);
            Frame frame = new Frame(frameIdx, kk, iter, framePhi, eta);
            sampledFrame = frame;
        }
        return sampledFrame;
    }

    /**
     * Sample topic assignment for each token.
     *
     * @param removeFromModel
     * @param addToModel
     * @param removeFromData
     * @param addToData
     * @param observe
     * @param extend
     * @return Elapsed time
     */
    protected long sampleZs_MH(
            boolean removeFromModel, boolean addToModel,
            boolean removeFromData, boolean addToData,
            boolean observe, boolean extend) {
        if (isReporting) {
            logln("+++ Sampling assignments ...");
        }
        long sTime = System.currentTimeMillis();
        numTokensChanged = 0;
        numFrameAssignmentChange = 0;
        numTokensAccepted = 0;

        for (int d = 0; d < D; d++) {
            int aa = authors[d];
            for (int n = 0; n < words[d].length; n++) {
                if (removeFromData) {
                    docTopics[d].decrement(t[d][n].topicIndex);
                    docFramesCounts[d][t[d][n].topicIndex].decrement(t[d][n].index);

                    auTopicCounts[aa].decrement(t[d][n].topicIndex);
                    auFrameCounts[aa][t[d][n].topicIndex].decrement(t[d][n].index);
                }

                if (removeFromModel) {
                    topics[t[d][n].topicIndex].phi.decrement(words[d][n]);
                    t[d][n].phi.decrement(words[d][n]);
                    if (t[d][n].phi.isEmpty()) {
                        topics[t[d][n].topicIndex].removeComponent(t[d][n].index);
                        topics[t[d][n].topicIndex].psi.remove(t[d][n].index);
                        t[d][n].changeToNew();

                        // debug
                        if (docFramesCounts[d][t[d][n].topicIndex].getCount(t[d][n].index) != 0) {
                            throw new RuntimeException();
                        }
                        if (auFrameCounts[aa][t[d][n].topicIndex].getCount(t[d][n].index) != 0) {
                            throw new RuntimeException();
                        }
                    }
                }

                Frame sampledFrame = sampleFrame(d, n, extend);
                boolean accept = false;
                if (t[d][n] == null) { // for initialization
                    accept = true;
                } else if (sampledFrame.equals(t[d][n])) {
                    accept = true;
                    numTokensAccepted++;
                } else {
                    Topic curTopic = topics[t[d][n].topicIndex];
                    Frame curFrame = t[d][n];
                    Topic newTopic = topics[sampledFrame.topicIndex];
                    Frame newFrame = sampledFrame;

                    double ratio = getMHRatio(d, n, curTopic, curFrame, newTopic, newFrame, observe);
                    if (rand.nextDouble() < ratio) {
                        accept = true;
                        numTokensAccepted++;
                    }
                }

                if (accept) { // accept the proposal move
                    if (t[d][n] != null) {
                        if (t[d][n].topicIndex != sampledFrame.topicIndex) {
                            numTokensChanged++;
                        }
                        if (!t[d][n].equals(sampledFrame)) {
                            numFrameAssignmentChange++;
                        }
                    }
                    t[d][n] = sampledFrame;
                }

                if (addToModel) {
                    topics[t[d][n].topicIndex].phi.increment(words[d][n]);
                    t[d][n].phi.increment(words[d][n]);
                }

                if (addToData) {
                    docTopics[d].increment(t[d][n].topicIndex);
                    docFramesCounts[d][t[d][n].topicIndex].increment(t[d][n].index);

                    auTopicCounts[aa].increment(t[d][n].topicIndex);
                    auFrameCounts[aa][t[d][n].topicIndex].increment(t[d][n].index);
                }

                // if a new frame is created, update psi and change the frame node status
                if (this.J == 0 && t[d][n].isNew) { // if accept
                    topics[t[d][n].topicIndex].createNewComponent(t[d][n].index, t[d][n]);
                    topics[t[d][n].topicIndex].psi.set(t[d][n].index,
                            frameAlphaGlobal / topics[t[d][n].topicIndex].phi.getCountSum()); // temp weight
                    topics[t[d][n].topicIndex].updatePsi();
                    t[d][n].changeToNormal();
                }
            }
        }

        long eTime = System.currentTimeMillis() - sTime;
        if (isReporting) {
            logln("--- --- time: " + eTime);
            logln("--- --- # tokens: " + numTokens
                    + ". # topic asgns: " + numTokensChanged
                    + " (" + MiscUtils.formatDouble((double) numTokensChanged / numTokens) + ")"
                    + ". # frame asgns: " + numFrameAssignmentChange
                    + " (" + MiscUtils.formatDouble((double) numFrameAssignmentChange / numTokens) + ")"
                    + ". # accepted: " + numTokensAccepted
                    + " (" + MiscUtils.formatDouble((double) numTokensAccepted / numTokens) + ")");

        }
        return eTime;
    }

    /**
     * Sample a frame node for a token in the proposal step of MH algorithm.
     *
     * @param dd Document index
     * @param nn Token index
     * @param extend
     */
    private Frame sampleFrame(int dd, int nn, boolean extend) {
        // sample topic
        double[] probs = new double[K];
        for (int kk = 0; kk < K; kk++) {
            probs[kk] = docTopics[dd].getProbability(kk) * topics[kk].getPhi(words[dd][nn]);
        }
        int kk = SamplerUtils.scaleSample(probs);
        Topic topic = topics[kk];

        // sample frame
        ArrayList<Frame> frameList = new ArrayList<>();
        ArrayList<Double> logprobList = new ArrayList<>();
        double norm = docFramesCounts[dd][kk].getCountSum() + frameAlphaLocal;

        // --- for existing frame node
        for (Frame frame : topic.getFrames()) {
            frameList.add(frame);
            int jj = frame.index;
            double pathprob = (docFramesCounts[dd][kk].getCount(jj)
                    + frameAlphaLocal * topic.psi.get(jj)) / norm;
            double wordprob = frame.getPhi(words[dd][nn]);
            logprobList.add(Math.log(pathprob * wordprob));
        }

        // --- for new frame node
        if (extend && this.J == 0) {
            frameList.add(null);
            double pathprob = frameAlphaLocal * topic.psi.get(NEW_CHILD_INDEX) / norm;
            double wordprob = uniform;
            logprobList.add(Math.log(pathprob * wordprob));
        }

        int sampledIdx = SamplerUtils.logMaxRescaleSample(logprobList);
        Frame sampledFrame = frameList.get(sampledIdx);
        if (sampledFrame == null) { // create new node
            assert this.J == 0;
            int frameIdx = topic.getNextIndex();
            double eta = SamplerUtils.getGaussian(0.0, gamma);
            DirMult framePhi = new DirMult(V, frameBeta * V, uniform);
            Frame frame = new Frame(frameIdx, kk, iter, framePhi, eta);
            sampledFrame = frame;
        }
        return sampledFrame;
    }

    /**
     * Compute the Metropolis-Hastings ratio.
     *
     * @param dd
     * @param nn
     * @param curTopic
     * @param curFrame
     * @param newTopic
     * @param newFrame
     */
    public double getMHRatio(int dd, int nn,
            Topic curTopic, Frame curFrame,
            Topic newTopic, Frame newFrame,
            boolean observe) {
        double newLogprob = getMHLogProb(dd, nn, newTopic, newFrame, observe);
        double curLogprob = getMHLogProb(dd, nn, curTopic, curFrame, observe);
        return Math.min(1.0, Math.exp(newLogprob - curLogprob));
    }

    /**
     * Get log probability to compute the Metropolis-Hasting ratio.
     */
    private double getMHLogProb(int dd, int nn, Topic topic, Frame frame, boolean observe) {
        int kk = topic.index;

        // response llh
        int aa = authors[dd];
        double resLlh = 0.0;
        if (observe) {
            double rawMean = 0.0;
            for (int jj : auFrameCounts[aa][kk].getIndices()) {
                if (topic.getFrame(jj) == null) {
                    System.out.println("topic frames: " + topic.getIndices());
                    for (int j : topic.getIndices()) {
                        System.out.println(">>> " + j + ". " + topic.getFrame(j).phi.getCountSum());
                    }
                    System.out.println("topic count sum: " + topic.phi.getCountSum());
                    System.out.println("author: " + auFrameCounts[aa][kk].getIndices());
                    for (int j : auFrameCounts[aa][kk].getIndices()) {
                        System.out.println(aa + ". " + j + ". " + auFrameCounts[aa][kk].getCount(j));
                    }
                    System.out.println("---> " + auTopicCounts[aa].getCount(kk)
                            + ". " + auFrameCounts[aa][kk].getCountSum());
                    System.out.println("--- " + topic.psi.toString());
                    System.out.println("--- " + auFrameCounts[aa][kk].getCount(jj));
                    throw new RuntimeException("");
                }

                rawMean += auFrameCounts[aa][kk].getCount(jj) * topic.getFrame(jj).eta;
            }
            double mean = (rawMean + frame.eta) / (auTopicCounts[aa].getCount(kk) + 1.0);
            resLlh = StatUtils.logNormalProbability(us[aa][kk], mean, sqrtRho);
        }

        // topic llh
        double topicLlh = topic.getPhi(words[dd][nn]);

        // frame llh
        double frameLlh = 0.0;
        double norm = docFramesCounts[dd][kk].getCountSum() + frameAlphaLocal;
        for (Frame child : topic.getFrames()) {
            int jj = child.index;
            double pathprob = (docFramesCounts[dd][kk].getCount(jj)
                    + frameAlphaLocal * topic.psi.get(jj)) / norm;
            double wordprob = child.getPhi(words[dd][nn]);
            frameLlh += pathprob * wordprob;
        }
        frameLlh = Math.log(frameLlh);

        double val = resLlh + frameLlh - topicLlh;
        return val;
    }

    /**
     * Update psi at all topic nodes.
     */
    protected long updatePsis() {
        if (isReporting) {
            logln("+++ Updating psi ...");
        }
        long sTime = System.currentTimeMillis();
        for (int kk = 0; kk < K; kk++) {
            topics[kk].updatePsi();
        }
        long eTime = System.currentTimeMillis() - sTime;
        if (isReporting) {
            logln("--- --- time: " + eTime);
        }
        return eTime;
    }

    /**
     * Update all eta vectors.
     */
    protected long updateEtas() {
        if (isReporting) {
            logln("+++ Updating etas ...");
        }
        long sTime = System.currentTimeMillis();

        int numConverged = 0;
        for (int kk = 0; kk < K; kk++) {
            boolean converged = updateEta(kk);
            if (converged) {
                numConverged++;
            }

            if (isReporting && !converged) {
                logln("*** not converged. Topic: " + kk
                        + ". # frames: " + topics[kk].getNumFrames()
                        + ". # tokens: " + topics[kk].phi.getCountSum());
            }
        }

        long eTime = System.currentTimeMillis() - sTime;
        if (isReporting) {
            logln("--- # converged: " + numConverged + " / " + K);
            logln("--- --- time: " + eTime);
        }
        return eTime;
    }

    /**
     * Update eta in a subtree.
     *
     * @param kk
     */
    private boolean updateEta(int kk) {
        int numFrames = topics[kk].getNumFrames();

        // create design matrix and responses
        ArrayList<SparseVector> designMatrixList = new ArrayList<>();
        ArrayList<Double> responseList = new ArrayList<>();
        for (int aa = 0; aa < A; aa++) {
            if (isBackedOff(aa, kk)) {
                continue;
            }
            responseList.add(us[aa][kk]);
            SparseVector vec = new SparseVector(numFrames);
            for (int ii = 0; ii < numFrames; ii++) {
                int jj = topics[kk].getSortedIndices().get(ii);
                double val = (double) auFrameCounts[aa][kk].getCount(jj)
                        / auFrameCounts[aa][kk].getCountSum();
                vec.set(ii, val);
            }
            designMatrixList.add(vec);
        }

        int numA = responseList.size();
        SparseVector[] designMatrix = new SparseVector[numA];
        double[] responses = new double[numA];
        for (int aa = 0; aa < numA; aa++) {
            designMatrix[aa] = designMatrixList.get(aa);
            responses[aa] = responseList.get(aa);
        }

        // current eta of this topic
        double[] topicEtas = new double[numFrames];
        for (int ii = 0; ii < numFrames; ii++) {
            int jj = topics[kk].getSortedIndices().get(ii);
            topicEtas[ii] = topics[kk].getFrame(jj).eta;
        }

        RidgeLinearRegressionOptimizable optimizable = new RidgeLinearRegressionOptimizable(
                responses, topicEtas, designMatrix, rho, 0.0, gamma);
        LimitedMemoryBFGS optimizer = new LimitedMemoryBFGS(optimizable);
        boolean converged = false;
        try {
            converged = optimizer.optimize();
        } catch (Exception ex) {
            ex.printStackTrace();
        }

        // update eta
        for (int ii = 0; ii < numFrames; ii++) {
            int jj = topics[kk].getSortedIndices().get(ii);
            Frame frame = topics[kk].getFrame(jj);
            if (converged) { // only update when convered
                frame.eta = optimizable.getParameter(ii);
            }
            if (iter > BURN_IN && iter % LAG == 0) { // store current eta
                frame.storeEta();
            }
        }

        return converged;
    }

    @Override
    protected long updateUXY() {
        if (isReporting) {
            logln("+++ Updating UXY ...");
        }
        long sTime = System.currentTimeMillis();
        rate = getAnnealingRate(epsilon / 10, iter, MAX_ITER);
        updateUs();
        updateXYs();
        long eTime = System.currentTimeMillis() - sTime;

        if (isReporting) {
            logln("--- --- time: " + eTime);
        }
        return eTime;
    }

    @Override
    protected double getLearningRate() {
        return this.epsilon;
    }

    @Override
    protected void updateUs() {
        Collections.shuffle(authorList);
        for (int aa : authorList) {
            if (!validAs[aa]) {
                continue;
            }
            Collections.shuffle(topicList);
            for (int kk : topicList) { // update 
                double llh = 0.0;
                for (int bb = 0; bb < B; bb++) {
                    if (isValidVote(aa, bb)) {
                        double dotprod = y[bb] + x[bb] * billThetas[bb].dotProduct(us[aa]);
                        double score = Math.exp(dotprod);
                        double prob = score / (1 + score);
                        llh += x[bb] * billThetas[bb].get(kk) * (getVote(aa, bb) - prob);
                    }
                }

                if (isBackedOff(aa, kk)) {
                    us[aa][kk] += (llh - (us[aa][kk] - u[aa]) / rho) * rate / B;
                } else {
                    us[aa][kk] += (llh - (us[aa][kk] - getLexicalU(aa, kk)) / rho) * rate / B;
                }
            }
        }
    }

    @Override
    public void updateXYs() {
        Collections.shuffle(billList);
        for (int bb : billList) {
            if (!validBs[bb]) {
                continue;
            }
            double llhX = 0.0;
            double llhY = 0.0;
            for (int aa = 0; aa < A; aa++) {
                if (isValidVote(aa, bb)) {
                    double dotprod = y[bb] + x[bb] * billThetas[bb].dotProduct(us[aa]);
                    double score = Math.exp(dotprod);
                    double prob = score / (1 + score);
                    llhX += (getVote(aa, bb) - prob) * billThetas[bb].dotProduct(us[aa]);
                    llhY += getVote(aa, bb) - prob;
                }
            }
            x[bb] += (llhX - x[bb] / sigma) * rate / A;
            y[bb] += (llhY - y[bb] / sigma) * rate / A;
        }
    }

    @Override
    public double getLogLikelihood() {
        return 0.0;
    }

    @Override
    public double getLogLikelihood(ArrayList<Double> newParams) {
        return 0.0;
    }

    @Override
    public void updateHyperparameters(ArrayList<Double> newParams) {

    }

    @Override
    public void validate(String msg) {
        for (int aa = 0; aa < A; aa++) {
            if (auTopicCounts[aa].getCountSum() != authorTotalWordWeights[aa]) {
                throw new MismatchRuntimeException(auTopicCounts[aa].getCountSum(),
                        (int) authorTotalWordWeights[aa]);
            }
            for (int kk = 0; kk < K; kk++) {
                if (auTopicCounts[aa].getCount(kk) != auFrameCounts[aa][kk].getCountSum()) {
                    throw new MismatchRuntimeException(auTopicCounts[aa].getCount(kk),
                            auFrameCounts[aa][kk].getCountSum());
                }
            }
        }

        for (int kk = 0; kk < K; kk++) {
            int count = 0;
            for (int dd = 0; dd < D; dd++) {
                count += docTopics[dd].getCount(kk);
            }
            if (count != topics[kk].phi.getCountSum()) {
                throw new MismatchRuntimeException(count, topics[kk].phi.getCountSum());
            }
        }

        for (int aa = 0; aa < A; aa++) {
            for (int kk = 0; kk < K; kk++) {
                for (int jj : auFrameCounts[aa][kk].getIndices()) {
                    if (topics[kk].getFrame(jj) == null) {
                        throw new RuntimeException("aa = " + aa
                                + ". kk = " + kk
                                + ". jj = " + jj
                                + ". count = " + auFrameCounts[aa][kk].getCount(jj)
                                + ". null frame.");
                    }
                }
            }
        }
    }

    private String getAssignmentString() {
        StringBuilder assignStr = new StringBuilder();
        for (int d = 0; d < D; d++) {
            assignStr.append(d).append("\n");
            assignStr.append(DirMult.output(docTopics[d])).append("\n");
            for (int kk = 0; kk < K; kk++) {
                assignStr.append(SparseCount.output(docFramesCounts[d][kk])).append("\n");
            }
            for (Frame tokenFrame : t[d]) {
                assignStr.append(tokenFrame.topicIndex).append("\t");
                assignStr.append(tokenFrame.index).append("\t");
            }
            assignStr.append("\n");
        }
        return assignStr.toString();
    }

    /**
     * Only keep old enough frame in the model.
     *
     * @param kk
     * @param jj
     */
    private boolean selectFrame(int kk, int jj) {
        Frame frame = topics[kk].getFrame(jj);
        return frame.born <= BURN_IN;
    }

    @Override
    public void outputState(String filepath) {
        if (verbose) {
            logln("--- Outputing current state to " + filepath);
        }

        try {
            // model
            StringBuilder modelStr = new StringBuilder();
            for (int kk = 0; kk < K; kk++) {
                Topic topic = topics[kk];
                modelStr.append(kk).append("\n");
                modelStr.append(DirMult.output(topic.phi)).append("\n");
                modelStr.append(SparseVector.output(topic.psi)).append("\n");

                ArrayList<Integer> selectedFrames = new ArrayList<>();
                for (int jj : topic.getIndices()) {
                    if (selectFrame(kk, jj)) {
                        selectedFrames.add(jj);
                    }
                }

                modelStr.append(selectedFrames.size()).append("\n");
                for (int jj : selectedFrames) {
                    Frame frame = topic.getFrame(jj);
                    modelStr.append(jj).append("\n");
                    modelStr.append(frame.born).append("\n");
                    modelStr.append(DirMult.output(frame.phi)).append("\n");
                    modelStr.append(frame.eta).append("\n");
                    modelStr.append(MiscUtils.listToString(frame.etaList)).append("\n");
                }
            }

            // train author scores
            StringBuilder authorStr = new StringBuilder();
            authorStr.append(A).append("\n");
            for (int aa = 0; aa < A; aa++) {
                authorStr.append(MiscUtils.arrayToString(us[aa])).append("\n");
            }

            StringBuilder authorSingleIPStr = new StringBuilder();
            if (u != null) {
                authorSingleIPStr.append(A).append("\n");
                for (int aa = 0; aa < A; aa++) {
                    authorSingleIPStr.append(u[aa]).append("\n");
                }
            }

            // train bill scores
            StringBuilder billStr = new StringBuilder();
            billStr.append(B).append("\n");
            for (int bb = 0; bb < B; bb++) {
                billStr.append(x[bb]).append("\n");
                billStr.append(y[bb]).append("\n");
                billStr.append(SparseVector.output(billThetas[bb])).append("\n");
                billStr.append(DirMult.output(billTopics[bb])).append("\n");
            }

            // output to a compressed file
            ArrayList<String> contentStrs = new ArrayList<>();
            contentStrs.add(modelStr.toString());
            contentStrs.add(getAssignmentString());
            contentStrs.add(billStr.toString());
            contentStrs.add(authorStr.toString());
            contentStrs.add(authorSingleIPStr.toString());

            String filename = IOUtils.removeExtension(IOUtils.getFilename(filepath));
            ArrayList<String> entryFiles = new ArrayList<>();
            entryFiles.add(filename + ModelFileExt);
            entryFiles.add(filename + AssignmentFileExt);
            entryFiles.add(filename + BillFileExt);
            entryFiles.add(filename + AuthorFileExt);
            entryFiles.add(filename + AuthorFileExt + ".singleip");

            this.outputZipFile(filepath, contentStrs, entryFiles);
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Excepion while outputing state to "
                    + filepath);
        }
    }

    @Override
    public void inputState(String filepath) {
        if (verbose) {
            logln("--- Reading state from " + filepath);
        }
        try {
            inputModel(filepath);
            inputAssignments(filepath);
            inputBillIdealPoints(filepath);
            inputAuthorIdealPoints(filepath);
            inputAuthorSingleIdealPoints(filepath);
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Excepion while inputing state from "
                    + filepath);
        }
    }

    public void inputModel(String zipFilepath) {
        if (verbose) {
            logln("--- --- Loading model from " + zipFilepath);
        }

        try {
            String filename = IOUtils.removeExtension(IOUtils.getFilename(zipFilepath));
            BufferedReader reader = IOUtils.getBufferedReader(zipFilepath,
                    filename + ModelFileExt);

            this.topics = new Topic[K];
            for (int kk = 0; kk < K; kk++) {
                int topicIdx = Integer.parseInt(reader.readLine());
                if (topicIdx != kk) {
                    throw new MismatchRuntimeException(topicIdx, kk);
                }
                DirMult topicPhi = DirMult.input(reader.readLine());
                SparseVector topicPsi = SparseVector.input(reader.readLine());
                this.topics[kk] = new Topic(kk, topicPhi);
                this.topics[kk].psi = topicPsi;
                this.topics[kk].phihat = topicPhi.getDistribution();

                int numFrames = Integer.parseInt(reader.readLine());
                for (int ii = 0; ii < numFrames; ii++) {
                    int jj = Integer.parseInt(reader.readLine());
                    int born = Integer.parseInt(reader.readLine());
                    DirMult framePhi = DirMult.input(reader.readLine());
                    double eta = Double.parseDouble(reader.readLine());
                    Frame frame = new Frame(jj, kk, born, framePhi, eta);
                    frame.etaList = MiscUtils.stringToList(reader.readLine());
                    frame.phihat = framePhi.getDistribution();
                    frame.changeToNormal();
                    this.topics[kk].createNewComponent(jj, frame);
                }
                this.topics[kk].fillInactives();
            }
            reader.close();
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while inputing model from "
                    + zipFilepath);
        }
    }

    public void inputAssignments(String zipFilepath) {
        if (verbose) {
            logln("--- --- Loading assignments from " + zipFilepath);
        }

        try {
            // initialize
            this.initializeDataStructure();

            String filename = IOUtils.removeExtension(IOUtils.getFilename(zipFilepath));
            BufferedReader reader = IOUtils.getBufferedReader(zipFilepath,
                    filename + AssignmentFileExt);

            for (int dd = 0; dd < D; dd++) {
                int docIdx = Integer.parseInt(reader.readLine().split("\t")[0]);
                if (docIdx != dd) {
                    throw new RuntimeException("Indices mismatch when loading assignments");
                }
                int aa = authors[dd];
                docTopics[dd] = DirMult.input(reader.readLine());
                for (int kk = 0; kk < K; kk++) {
                    docFramesCounts[dd][kk] = SparseCount.input(reader.readLine());
                }

                String[] sline = reader.readLine().split("\t");
                if (sline.length != words[dd].length * 2) {
                    throw new RuntimeException("[MISMATCH]. Doc "
                            + dd + ". " + sline.length + " vs. " + words[dd].length * 2);
                }
                for (int nn = 0; nn < words[dd].length; nn++) {
                    int kk = Integer.parseInt(sline[2 * nn]);
                    int jj = Integer.parseInt(sline[2 * nn + 1]);
                    t[dd][nn] = topics[kk].getFrame(jj);
                    auTopicCounts[aa].increment(kk);
                    auFrameCounts[aa][kk].increment(jj);
                }
            }
            reader.close();
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while inputing assignments from "
                    + zipFilepath);
        }
    }

    /**
     * Input author ideal points.
     *
     * @param zipFilepath File path
     */
    @Override
    public void inputAuthorIdealPoints(String zipFilepath) {
        if (verbose) {
            logln("--- --- Loading author scores from " + zipFilepath);
        }
        try {
            String filename = IOUtils.removeExtension(IOUtils.getFilename(zipFilepath));
            BufferedReader reader = IOUtils.getBufferedReader(zipFilepath, filename + AuthorFileExt);
            int numAuthors = Integer.parseInt(reader.readLine());
            if (numAuthors != A) {
                throw new MismatchRuntimeException(numAuthors, A);
            }
            us = new double[A][];
            for (int aa = 0; aa < A; aa++) {
                us[aa] = MiscUtils.stringToDoubleArray(reader.readLine());
            }
            reader.close();
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while loading model from "
                    + zipFilepath);
        }
    }

    public void inputAuthorSingleIdealPoints(String zipFilepath) {
        if (verbose) {
            logln("--- --- Loading author scores from " + zipFilepath);
        }
        try {
            String filename = IOUtils.removeExtension(IOUtils.getFilename(zipFilepath));
            BufferedReader reader = IOUtils.getBufferedReader(zipFilepath, filename + AuthorFileExt + ".singleip");
            String line = reader.readLine();
            if (line.isEmpty()) {
                return;
            }
            int numAuthors = Integer.parseInt(line);
            if (numAuthors != A) {
                throw new MismatchRuntimeException(numAuthors, A);
            }
            u = new double[A];
            for (int aa = 0; aa < A; aa++) {
                u[aa] = Double.parseDouble(reader.readLine());
            }
            reader.close();
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while loading model from "
                    + zipFilepath);
        }
    }

    /**
     * Input bill ideal points.
     *
     * @param zipFilepath File path
     */
    @Override
    public void inputBillIdealPoints(String zipFilepath) {
        if (verbose) {
            logln("--- --- Loading bill scores from " + zipFilepath);
        }
        try {
            String filename = IOUtils.removeExtension(IOUtils.getFilename(zipFilepath));
            BufferedReader reader = IOUtils.getBufferedReader(zipFilepath, filename + BillFileExt);

            int numBills = Integer.parseInt(reader.readLine());
            if (numBills != B) {
                throw new MismatchRuntimeException(numBills, B);
            }
            x = new double[B];
            y = new double[B];
            billTopics = new DirMult[B];
            billThetas = new SparseVector[B];
            for (int bb = 0; bb < B; bb++) {
                x[bb] = Double.parseDouble(reader.readLine());
                y[bb] = Double.parseDouble(reader.readLine());
                billThetas[bb] = SparseVector.input(reader.readLine());
                billTopics[bb] = DirMult.input(reader.readLine());
            }
            reader.close();
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while loading model from "
                    + zipFilepath);
        }
    }

    @Override
    public void outputTopicTopWords(File file, int numTopWords) {
        this.outputTopicTopWords(file, numTopWords, topicVocab);
    }

    public void outputTopicTopWords(File file, int numTopWords, ArrayList<String> topicLabels) {
        if (this.wordVocab == null) {
            throw new RuntimeException("The word vocab has not been assigned yet");
        }

        if (verbose) {
            logln("Outputing per-topic top words to " + file);
        }

        try {
            BufferedWriter writer = IOUtils.getBufferedWriter(file);
            for (int kk = 0; kk < K; kk++) {
                String topicLabel = "Topic " + kk;
                if (topicLabels != null) {
                    topicLabel += ": " + topicLabels.get(kk);
                }

                // count from documents
                int count = 0;
                for (int dd = 0; dd < D; dd++) {
                    count += docTopics[dd].getCount(kk);
                }

                double[] distrs = topics[kk].getPhi();
                String[] topWords = getTopWords(distrs, numTopWords);
                writer.write("[" + topicLabel
                        + ", " + topics[kk].phi.getCountSum()
                        + ", " + count
                        + "]");
                for (String topWord : topWords) {
                    writer.write("\t" + topWord);
                }
                writer.write("\n\n");

                // frames
                ArrayList<RankingItem<Integer>> sortedFrames = new ArrayList<RankingItem<Integer>>();
                for (Frame frame : topics[kk].getFrames()) {
                    sortedFrames.add(new RankingItem<Integer>(frame.index, frame.eta));
                }
                Collections.sort(sortedFrames);

                for (RankingItem<Integer> sortedFrame : sortedFrames) {
                    int jj = sortedFrame.getObject();
                    Frame frame = topics[kk].getFrame(jj);

                    if (!selectFrame(kk, jj)) {
                        continue;
                    }

                    String frameLabel = "Frame " + kk + ":" + jj;
                    distrs = frame.getPhi();
                    topWords = getTopWords(distrs, numTopWords);

                    // count from document
                    count = 0;
                    for (int dd = 0; dd < D; dd++) {
                        count += docFramesCounts[dd][kk].getCount(jj);
                    }

                    writer.write("\t[" + frameLabel
                            + ", " + frame.born
                            + ", " + frame.phi.getCountSum()
                            + ", " + count
                            + ", " + MiscUtils.formatDouble(topics[kk].psi.get(jj))
                            + ", " + MiscUtils.formatDouble(frame.eta)
                            + "] "
                            + MiscUtils.listToString(frame.etaList)
                            + " ["
                            + "m: " + MiscUtils.formatDouble(StatUtils.mean(frame.etaList))
                            + ", s: " + MiscUtils.formatDouble(StatUtils.standardDeviation(frame.etaList))
                            + "]"
                            + "\n");
                    for (String topWord : topWords) {
                        writer.write("\t" + topWord);
                    }
                    writer.write("\n\n");
                }
                writer.write("\n\n");
            }
            writer.close();
        } catch (IOException e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while outputing topic words");
        }
    }

    private String getTopicLabel(int kk) {
        String label = "Topic-" + kk;
        if (topicVocab != null) {
            label += ": " + topicVocab.get(kk);
        }
        return label;
    }

    public void printGlobalTreeSummery() {
        int totalNumFrames = 0;
        for (int kk = 0; kk < K; kk++) {
            totalNumFrames += topics[kk].getNumFrames();
        }
        logln("# frames: " + totalNumFrames);
        logln("# frames per topic: " + (double) totalNumFrames / K);
    }

    /**
     * The current tree.
     *
     * @return The current tree
     */
    public void printGlobalTree() {
        int totalNumFrames = 0;
        for (int kk = 0; kk < K; kk++) {
            // topic
            double[] distrs = topics[kk].phi.getDistribution();
            String[] topWords = getTopWords(distrs, 15);
            System.out.println(getTopicLabel(kk)
                    + " [" + topics[kk].phi.getCountSum()
                    + ", " + topics[kk].getNumFrames()
                    + "]");
            for (String topWord : topWords) {
                System.out.print(topWord + " ");
            }
            System.out.println("\n");

            // frames
            ArrayList<RankingItem<Integer>> sortedFrames = new ArrayList<RankingItem<Integer>>();
            for (Frame frame : topics[kk].getFrames()) {
                sortedFrames.add(new RankingItem<Integer>(frame.index, frame.eta));
            }
            Collections.sort(sortedFrames);

            for (RankingItem<Integer> sortedFrame : sortedFrames) {
                int jj = sortedFrame.getObject();
                Frame frame = topics[kk].getFrame(jj);
                String frameLabel = "Frame " + kk + ":" + jj;
                distrs = frame.phi.getDistribution();
                topWords = getTopWords(distrs, 15);

                System.out.println("\t[" + frameLabel
                        + ", " + frame.phi.getCountSum()
                        + ", " + MiscUtils.formatDouble(frame.eta)
                        + "]");
                System.out.print("\t");
                for (String topWord : topWords) {
                    System.out.print(topWord + " ");
                }
                System.out.println("\n");
            }
            System.out.println("\n");

            totalNumFrames += sortedFrames.size();
        }
        logln("# frames: " + totalNumFrames);
        logln("# frames per topic: " + (double) totalNumFrames / K);
    }

    ////////////////////////////////////////////////////////////////////////////
    public void outputHTMLFile(File outputFile,
            ArrayList<Integer> docIndices,
            String[] docIds,
            String[][] docSentRawTexts,
            HashMap<String, Author> authorTable,
            String govtrackUrl) {
        if (verbose) {
            logln("Outputing to HTML file " + outputFile);
        }
        StringBuilder str = new StringBuilder();
        str.append("<table>\n");
        str.append("<tbody>\n");

        for (int kk = 0; kk < K; kk++) {
            // first-level topics
            Topic topic = topics[kk];
            String firstLevelColor = "#FFFF99";
            str.append("<tr class=\"level2\"><td style=\"background-color:#FFFFFF\"></td></tr>\n");
            str.append("<tr class=\"level1").append("\">\n");
            str.append("<td style=\"background-color:").append(firstLevelColor)
                    .append(";color:").append(HTMLUtils.getTextColor(firstLevelColor))
                    .append(";\"").append(">\n")
                    .append("[")
                    .append(topicVocab.get(kk))
                    .append("]</a>")
                    .append(" (").append(topic.phi.getCountSum())
                    .append(")");
            double[] distrs = topic.phi.getDistribution();
            String[] topWords = getTopWords(distrs, 15);
            for (String word : topWords) {
                str.append(" ").append(word);
            }
            str.append("</td>\n");
            str.append("</tr>\n");

            // second-level frames
            ArrayList<RankingItem<Integer>> sortedFrames = new ArrayList<RankingItem<Integer>>();
            for (Frame frame : topics[kk].getFrames()) {
                sortedFrames.add(new RankingItem<Integer>(frame.index, frame.eta));
            }
            Collections.sort(sortedFrames);

            for (RankingItem<Integer> sortedFrame : sortedFrames) {
                int jj = sortedFrame.getObject();
                Frame frame = topics[kk].getFrame(jj);
                String frameLabel = "Frame " + kk + ":" + jj;
                distrs = frame.phi.getDistribution();
                topWords = getTopWords(distrs, 15);

                String color = HTMLUtils.WHITE;
                String framePathStr = "0:" + kk + ":" + jj;
                str.append("<tr class=\"level2\"><td style=\"background-color:#FFFFFF\"></td></tr>\n");
                str.append("<tr class=\"level2").append("\">\n");
                str.append("<td style=\"background-color:").append(color)
                        .append(";color:").append(HTMLUtils.getTextColor(color))
                        .append(";\"").append(">\n")
                        .append("<a style=\"text-decoration:underline;color:blue;\" onclick=\"showme('")
                        .append(framePathStr)
                        .append("');\" id=\"toggleDisplay\">")
                        .append("[")
                        .append(framePathStr)
                        .append("]</a>")
                        .append(" (")
                        .append(frameLabel)
                        .append(": ").append(frame.phi.getCountSum())
                        .append(", ").append(MiscUtils.formatDouble(frame.eta))
                        .append(")");
                for (String word : topWords) {
                    str.append(" ").append(word);
                }
                str.append("</td>\n");
                str.append("</tr>\n");

                // snippets
                ArrayList<RankingItem<Integer>> rankDocs = rankDocuments(frame);
                str.append("<tr class=\"level2").append("\"")
                        .append(" id=\"").append(framePathStr).append("\"")
                        .append(" style=\"display:none;\"")
                        .append(">\n");
                str.append("<td>\n");
                str.append("<ol>\n");
                for (int i = 0; i < Math.min(20, rankDocs.size()); i++) {
                    RankingItem<Integer> rankDoc = rankDocs.get(i);
                    int ii = rankDoc.getObject();
                    int dd = docIndices.get(ii);

                    // doc id
                    String debateId = docIds[dd].split("_")[0];
                    str.append("<li>[")
                            .append(HTMLUtils.getEmbeddedLink(debateId,
                                            govtrackUrl + debateId + ".xml"))
                            .append("] ");
                    str.append("</li>\n");

                    ArrayList<String> docInfoList = new ArrayList<>();

                    // doc text
                    StringBuilder docStr = new StringBuilder();
                    for (String ss : docSentRawTexts[dd]) {
                        docStr.append(ss).append(" ");
                    }
                    docInfoList.add(docStr.toString());

                    int aIdx = authors[ii];
                    String authorId = authorVocab.get(aIdx);
                    String authorName = authorTable.get(authorId).getProperty(GTLegislator.NAME);
                    String authorFWID = authorTable.get(authorId).getProperty(GTLegislator.FW_ID);
                    docInfoList.add(HTMLUtils.getEmbeddedLink(authorName,
                            "http://congress.freedomworks.org/node/" + authorFWID));
                    String authorParty = authorTable.get(authorId).getProperty(GTLegislator.PARTY);
                    docInfoList.add(authorParty);
                    String nominateScore = authorTable.get(authorId).getProperty(GTLegislator.NOMINATE_SCORE1);
                    docInfoList.add("DW-NOMINATE Score: " + nominateScore);
//                    String fwScore = authorTable.get(authorId).getProperty(GTLegislator.FW_SCORE);
//                    docInfoList.add("FW Score: " + fwScore);
//                    docInfoList.add("Estmated score: " + u[authors[ii]]);
                    str.append(HTMLUtils.getHTMLList(docInfoList)).append("\n");
                }
                str.append("</ol>\n");
                str.append("</td>\n");
                str.append("</tr>\n");
//            }
            }
        }
        str.append("</tbody>\n");
        str.append("</table>\n");
        HTMLUtils.outputHTMLFile(outputFile, str.toString());
    }

    private ArrayList<RankingItem<Integer>> rankDocuments(Frame node) {
        ArrayList<RankingItem<Integer>> rankDocs = new ArrayList<>();
        for (int dd = 0; dd < D; dd++) {
            if (words[dd].length < 10) {
                continue;
            }
            double count = docFramesCounts[dd][node.topicIndex].getCount(node.index);
            double val = (double) count / words[dd].length;
            rankDocs.add(new RankingItem<Integer>(dd, val));
        }
        Collections.sort(rankDocs);
        return rankDocs;
    }

    public void outputTopicAttentions(File outputFile,
            HashMap<Integer, Boolean> teapartycaucus) {
        if (verbose) {
            logln("Output to " + outputFile);
        }
        try {
            BufferedWriter writer = IOUtils.getBufferedWriter(outputFile);
            SparseCount tp = new SparseCount();
            SparseCount ntp = new SparseCount();
            for (int dd = 0; dd < D; dd++) {
                int aa = authors[dd];
                int aidx = this.authorIndices.get(aa);
                Boolean tpc = teapartycaucus.get(aidx);
                if (tpc) {
                    for (int kk : docTopics[dd].getCounts()) {
                        int count = docTopics[dd].getCount(kk);
                        tp.changeCount(kk, count);
                    }
                } else {
                    for (int kk : docTopics[dd].getCounts()) {
                        int count = docTopics[dd].getCount(kk);
                        ntp.changeCount(kk, count);
                    }
                }
            }

            writer.write("Issues\tTea Party Caucus\tNon-Tea Party Caucus\n");
            for (int kk = 0; kk < K; kk++) {
                writer.write("\"" + topicVocab.get(kk).replaceAll(",", " ") + "\""
                        + "\t" + tp.getCount(kk)
                        + "\t" + ntp.getCount(kk)
                        + "\n");
            }

            writer.close();
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while outputing to " + outputFile);
        }
    }

    public void outputTheta(File outputFile) {
        if (verbose) {
            logln("Output thetas to " + outputFile);
        }
        try {
            BufferedWriter writer = IOUtils.getBufferedWriter(outputFile);
            writer.write("Topic\tCount\n");
            for (int kk = 0; kk < K; kk++) {
                writer.write(getTopicLabel(kk) + "\t" + topics[kk].phi.getCountSum() + "\n");
            }
            writer.close();
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while outputing to " + outputFile);
        }
    }

    private boolean isSelected(Topic topic, Frame frame) {
        if (frame.born > MAX_ITER - 10) {
            return false;
        }
        return topic.psi.get(frame.index) >= 0.1;
    }

    private double getScore(ArrayList<Integer> topWords, int word) {
        int idx = topWords.indexOf(word);
        if (idx < 0) {
            return 0.0;
        } else {
            return 1.0 / (idx + 1);
        }
    }

    public void outputHierarchyWithDetails(File outputFile,
            File outputSubtreeFolder,
            HashMap<String, Author> authorTable,
            String[] docIds,
            String[][] docRawSentences) {
        if (this.wordVocab == null) {
            throw new RuntimeException("The word vocab has not been assigned yet");
        }

        if (verbose) {
            logln("Outputing per-topic top words to " + outputFile);
        }

        // validate
        for (int aa = 0; aa < A; aa++) {
            int authorTokens = 0;
            for (int kk = 0; kk < K; kk++) {
                authorTokens += auFrameCounts[aa][kk].getCountSum();
            }
            if (authorTokens != authorTotalWordWeights[aa]) {
                throw new MismatchRuntimeException(authorTokens, (int) authorTotalWordWeights[aa]);
            }
        }

        if (this.authorVocab.size() != A) {
            throw new MismatchRuntimeException(this.authorVocab.size(), A);
        }

        ArrayList<Integer>[] authorDocIndices = new ArrayList[A];
        for (int aa = 0; aa < A; aa++) {
            authorDocIndices[aa] = new ArrayList<>();
        }
        for (int dd = 0; dd < D; dd++) {
            authorDocIndices[authors[dd]].add(dd);
        }

        try {
            BufferedWriter writer = IOUtils.getBufferedWriter(outputFile);
            BufferedWriter textWriter = IOUtils.getBufferedWriter(outputFile + ".text");

            ArrayList<RankingItem<Integer>> rankIssues = new ArrayList<>();
            for (int kk = 0; kk < K; kk++) {
                ArrayList<Double> frameEtaList = new ArrayList<>();
                for (Frame frame : topics[kk].getFrames()) {
                    if (isSelected(topics[kk], frame)) {
                        frameEtaList.add(StatUtils.mean(frame.etaList));
                    }
                }
                Collections.sort(frameEtaList);
                double diff = frameEtaList.get(0) - frameEtaList.get(frameEtaList.size() - 1);
                rankIssues.add(new RankingItem<Integer>(kk, diff));
            }
            Collections.sort(rankIssues);

            for (int i = 0; i < K; i++) {
                RankingItem<Integer> rankIssue = rankIssues.get(i);
                int kk = rankIssue.getObject();

                String issueLabel = "issue_" + kk + "_"
                        + topicVocab.get(kk).replaceAll(" ", "")
                        .replaceAll(",", "")
                        .replaceAll("\\.", "");
                BufferedWriter subtreeWriter = IOUtils.getBufferedWriter(
                        new File(outputSubtreeFolder, issueLabel));
                BufferedWriter subtreeTextWriter = IOUtils.getBufferedWriter(
                        new File(outputSubtreeFolder, issueLabel + ".text"));

                double[] distrs = topics[kk].phi.getDistribution();
                String[] topWords = getTopWords(distrs, 15);
                writer.write("[" + getTopicLabel(kk)
                        + ", " + topics[kk].phi.getCountSum()
                        + ", " + MiscUtils.formatDouble(rankIssue.getPrimaryValue())
                        + "] ");
                for (String topWord : topWords) {
                    writer.write(topWord + ", ");
                }
                writer.write("\n\n");

                // subtree
                subtreeWriter.write("[" + getTopicLabel(kk)
                        + ", " + topics[kk].phi.getCountSum()
                        + ", " + MiscUtils.formatDouble(rankIssue.getPrimaryValue())
                        + "] ");
                for (String topWord : topWords) {
                    subtreeWriter.write(topWord + ", ");
                }
                subtreeWriter.write("\n\n");

                // frames
                ArrayList<RankingItem<Integer>> sortedFrames = new ArrayList<RankingItem<Integer>>();
                for (Frame frame : topics[kk].getFrames()) {
                    if (!isSelected(topics[kk], frame)) {
                        continue;
                    }
                    double meanEta = StatUtils.mean(frame.etaList);
                    sortedFrames.add(new RankingItem<Integer>(frame.index, meanEta));
                }
                Collections.sort(sortedFrames);

                for (RankingItem<Integer> sortedFrame : sortedFrames) {
                    int jj = sortedFrame.getObject();
                    Frame frame = topics[kk].getFrame(jj);
                    ArrayList<RankingItem<Integer>> rankAuthors = new ArrayList<>();
                    for (int aa = 0; aa < A; aa++) {
                        double val = 0.0;
                        if (!isBackedOff(aa, kk)) {
                            val = (double) auFrameCounts[aa][kk].getCount(jj)
                                    / auFrameCounts[aa][kk].getCountSum();
                        }
                        rankAuthors.add(new RankingItem<Integer>(aa, val));
                    }
                    Collections.sort(rankAuthors);
                    String frameLabel = "Frame " + kk + ":" + jj;
                    distrs = topics[kk].getFrame(jj).phi.getDistribution();
                    topWords = getTopWords(distrs, 15);
                    // whole tree
                    writer.write("\t[" + frameLabel
                            + ", idx: " + frame.index
                            + ", b: " + frame.born
                            + ", psi: " + MiscUtils.formatDouble(topics[kk].psi.get(jj))
                            + ", count: " + topics[kk].getFrame(jj).phi.getCountSum()
                            + ", e: " + MiscUtils.formatDouble(topics[kk].getFrame(jj).eta)
                            + ", m: " + MiscUtils.formatDouble(StatUtils.mean(frame.etaList))
                            + ", s: " + MiscUtils.formatDouble(StatUtils.standardDeviation(frame.etaList))
                            + "]"
                            + " " + MiscUtils.listToString(frame.etaList));
                    writer.write("\n\t");
                    for (String topWord : topWords) {
                        writer.write(", " + topWord);
                    }
                    writer.write("\n");
                    // subtrees
                    subtreeWriter.write("\t[" + frameLabel
                            + ", " + MiscUtils.formatDouble(topics[kk].psi.get(jj))
                            + ", " + topics[kk].getFrame(jj).phi.getCountSum()
                            + ", " + MiscUtils.formatDouble(topics[kk].getFrame(jj).eta)
                            + ", " + MiscUtils.formatDouble(StatUtils.mean(frame.etaList))
                            + ", " + MiscUtils.formatDouble(StatUtils.standardDeviation(frame.etaList))
                            + "]"
                            + " " + MiscUtils.listToString(frame.etaList));
                    subtreeWriter.write("\n\t");
                    for (String topWord : topWords) {
                        subtreeWriter.write(topWord + ", ");
                    }
                    subtreeWriter.write("\n");

                    subtreeTextWriter.write("\t[" + frameLabel
                            + ", " + MiscUtils.formatDouble(topics[kk].psi.get(jj))
                            + ", " + topics[kk].getFrame(jj).phi.getCountSum()
                            + ", " + MiscUtils.formatDouble(topics[kk].getFrame(jj).eta)
                            + ", " + MiscUtils.formatDouble(StatUtils.mean(frame.etaList))
                            + ", " + MiscUtils.formatDouble(StatUtils.standardDeviation(frame.etaList))
                            + "]"
                            + " " + MiscUtils.listToString(frame.etaList));
                    subtreeTextWriter.write("\n\t");
                    for (String topWord : topWords) {
                        subtreeTextWriter.write(topWord + ", ");
                    }
                    subtreeTextWriter.write("\n");

                    // rank doc by matching top words
                    ArrayList<Integer> topWordIndices = new ArrayList<>();
                    ArrayList<RankingItem<Integer>> rankWs = new ArrayList<>();
                    for (int vv = 0; vv < V; vv++) {
                        rankWs.add(new RankingItem<Integer>(vv, distrs[vv]));
                    }
                    Collections.sort(rankWs);
                    for (int yy = 0; yy < 15; yy++) {
                        topWordIndices.add(rankWs.get(yy).getObject());
                    }

                    // documents
                    for (int xx = 0; xx < Math.min(10, rankAuthors.size()); xx++) {
                        RankingItem<Integer> rankAuthor = rankAuthors.get(xx);
                        int aa = rankAuthor.getObject();
                        String authorId = authorVocab.get(aa);
                        Author author = authorTable.get(authorId);

                        subtreeWriter.write("\t\t" + aa
                                + ". [ID: " + authorId
                                + ", " + author.getProperty(GTLegislator.NAME)
                                + ", DW: " + author.getProperty(GTLegislator.NOMINATE_SCORE1)
                                + ", us: " + MiscUtils.formatDouble(us[aa][kk])
                                + ", u: " + MiscUtils.formatDouble(u[aa])
                                + "] ["
                                + MiscUtils.formatDouble(rankAuthor.getPrimaryValue())
                                + ", " + auFrameCounts[aa][kk].getCount(jj)
                                + ", " + auFrameCounts[aa][kk].getCountSum()
                                + ", " + authorTotalWordWeights[aa]
                                + "]"
                                + "\n");
                        // rank document
                        ArrayList<RankingItem<Integer>> rankDocs = new ArrayList<>();
                        for (int dd : authorDocIndices[aa]) {
                            int count = docFramesCounts[dd][kk].getCount(jj);
                            if (count > 0) {
                                rankDocs.add(new RankingItem<Integer>(dd, (double) count));
                            }
                        }
                        Collections.sort(rankDocs);

                        for (int yy = 0; yy < Math.min(5, rankDocs.size()); yy++) {
                            RankingItem<Integer> rankDoc = rankDocs.get(yy);
                            int dd = rankDoc.getObject();
                            subtreeWriter.write("\t\t\t" + dd
                                    + "\t" + docIds[docIndices.get(dd)]
                                    + ". " + rankDoc.getPrimaryValue()
                                    + ". " + words[dd].length
                                    + "\n");

                            StringBuilder docStr = new StringBuilder();
                            for (String ss : docRawSentences[docIndices.get(dd)]) {
                                docStr.append(ss).append(" ");
                            }
                            textWriter.write(kk + ", " + jj
                                    + ", " + dd
                                    + "\t" + docIds[docIndices.get(dd)]
                                    + "\t" + words[dd].length
                                    + "\t" + docStr.toString() + "\n\n");

//                            subtreeTextWriter.write(kk + ", " + jj
//                                    + ", " + dd
//                                    + "\t" + docIds[docIndices.get(dd)]
//                                    + "\t" + words[dd].length
//                                    + "\t" + docStr.toString() + "\n\n");
                        }

                        // rank document according to its words matching the top words of the frame
                        ArrayList<RankingItem<Integer>> rankMatchDocs = new ArrayList<>();
                        for (int dd : authorDocIndices[aa]) {
                            int count = docFramesCounts[dd][kk].getCount(jj);
                            if (count < 0) {
                                continue;
                            }

                            double score = 0.0;
                            for (int nn = 0; nn < words[dd].length; nn++) {
                                score += getScore(topWordIndices, words[dd][nn]);
                            }
                            rankMatchDocs.add(new RankingItem<Integer>(dd, score));
                        }
                        Collections.sort(rankMatchDocs);

                        for (int yy = 0; yy < Math.min(10, rankMatchDocs.size()); yy++) {
                            RankingItem<Integer> rankDoc = rankMatchDocs.get(yy);
                            int dd = rankDoc.getObject();
                            subtreeTextWriter.write("\t\t"
                                    + "[" + kk + ", " + jj + "]"
                                    + "[ aid: " + authorId
                                    + ", " + author.getProperty(GTLegislator.NAME)
                                    + ", us: " + MiscUtils.formatDouble(us[aa][kk])
                                    + ", u: " + MiscUtils.formatDouble(u[aa]) + "]"
                                    + "[dd: " + dd
                                    + ", val: " + MiscUtils.formatDouble(rankDoc.getPrimaryValue())
                                    + ", " + docIds[docIndices.get(dd)] + "]"
                                    + "\t" + words[dd].length + "\n");
                        }
                        subtreeTextWriter.write("\n");
                    }
                    writer.write("\n\n");
                }
                writer.write("\n\n");

                subtreeWriter.close();
                subtreeTextWriter.close();
            }
            writer.close();
            textWriter.close();
        } catch (IOException e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while outputing topic words");
        }
    }

    public void outputAuthorDetails(HashMap<String, Author> authorTable) {
        File authorFolder = new File(this.getSamplerFolderPath(), "authors");
        IOUtils.createFolder(authorFolder);

        try {
            for (int aa = 0; aa < A; aa++) {
                String authorId = this.authorVocab.get(aa);
                Author author = authorTable.get(authorId);

                BufferedWriter writer = IOUtils.getBufferedWriter(new File(authorFolder, authorId));
                for (int kk = 0; kk < K; kk++) {
                    writer.write(kk
                            + "\t" + topicVocab.get(kk)
                            + "\t" + us[aa][kk]
                            + "\n");
                }

//                writer.write("Id: " + authorId + "\n");
//                writer.write("Name: " + author.getProperty(GTLegislator.NAME) + "\n");
//                writer.write("ICPSR ID: " + author.getProperty(GTLegislator.ICPSRID) + "\n");
//                writer.write("FreedomWorks ID: " + author.getProperty(GTLegislator.FW_ID) + "\n");
//                writer.write("Ideal point: " + u[aa] + "\n");
//                writer.write("# tokens: " + auTopicCounts[aa].getCountSum() + "\n");
//
//                for (int kk = 0; kk < K; kk++) {
//                    writer.write("kk = " + kk + ": " + topicVocab.get(kk) + "\n");
//                    writer.write("\t" + us[aa][kk] + "\n");
//                    writer.write("\t" + auTopicCounts[aa].getCount(kk) + "\n");
//                    for (int jj : auFrameCounts[aa][kk].getSortedIndices()) {
//                        writer.write("\t\tjj = " + jj + "\t" + topics[kk].getFrame(jj).eta + "\t" + auFrameCounts[aa][kk].getCount(jj) + "\n");
//                    }
//                    writer.write("\n");
//                }
                writer.close();
            }
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while outputing to " + authorFolder);
        }
    }
    ////////////////////////////////////////////////////////////////////////////

    /**
     * Average author features from multiple samples.
     *
     * @param partialModelFolder
     * @param partialAssignmentFolder
     */
    public SparseVector[] getAuthorFeatures(File partialModelFolder, File partialAssignmentFolder) {
        SparseVector[] authorFeatures = new SparseVector[A];

        String[] filenames = partialModelFolder.list();
        int count = 0;
        for (String filename : filenames) {
            if (!filename.endsWith(".zip")) {
                continue;
            }

            File partModelFile = new File(partialModelFolder, filename);
            File partAssgnFile = new File(partialAssignmentFolder, filename);

            if (!partModelFile.exists() || !partAssgnFile.exists()) {
                continue;
            }

            inputBillIdealPoints(partModelFile.getAbsolutePath());
            inputModel(partModelFile.getAbsolutePath());
            inputAuthorIdealPoints(partAssgnFile.getAbsolutePath());
            inputAssignments(partAssgnFile.getAbsolutePath());
            inputAuthorSingleIdealPoints(partAssgnFile.getAbsolutePath());

            SparseVector[] partAuthorFeatures = getSingleModelAuthorFeatures();
            for (int aa = 0; aa < A; aa++) {
                if (authorFeatures[aa] == null) {
                    authorFeatures[aa] = partAuthorFeatures[aa];
                } else {
                    for (int ff : partAuthorFeatures[aa].getIndices()) {
                        authorFeatures[aa].change(ff, partAuthorFeatures[aa].get(ff));
                    }
                }
            }
            count++;
        }

        for (int aa = 0; aa < A; aa++) {
            authorFeatures[aa].scale(1.0 / count);
        }

        return authorFeatures;
    }

    private SparseVector[] getSingleModelAuthorFeatures() {
        SparseVector[] authorFeatures = new SparseVector[A];
        for (int aa = 0; aa < A; aa++) {
            int count = 0;
            authorFeatures[aa] = new SparseVector();

            if (u != null) {
                for (int bb = 0; bb < B; bb++) {
                    double dotprod = y[bb];
                    for (int kk = 0; kk < K; kk++) {
                        dotprod += x[bb] * billThetas[bb].get(kk) * us[aa][kk];
                    }
                    double score = Math.exp(dotprod);
                    double prob = score / (1.0 + score);
                    authorFeatures[aa].set(count++, prob);
//                    authorFeatures[aa].set(count++, (double)getVote(aa, bb));
                }
            }
            for (int bb = 0; bb < B; bb++) {
                double avgU = 0.0;
                int c = 0;
                for (int kk = 0; kk < K; kk++) {
                    if (!isBackedOff(aa, kk)) {
                        avgU += us[aa][kk];
                        c++;
                    }
                }
                if (c != 0) {
                    avgU /= c;
                }
                double dotprod = y[bb] + x[bb] * avgU;
                double score = Math.exp(dotprod);
                double prob = score / (1.0 + score);
                authorFeatures[aa].set(count++, prob);
            }

            for (int kk = 0; kk < K; kk++) {
                authorFeatures[aa].set(count++, us[aa][kk]);
            }

            for (int kk = 0; kk < K; kk++) {
                double val = 0.0;
                if (auTopicCounts[aa].getCountSum() != 0) {
                    val = (double) auTopicCounts[aa].getCount(kk) / auTopicCounts[aa].getCountSum();
                }
                authorFeatures[aa].set(count++, val);
            }

            for (int kk = 0; kk < K; kk++) {
                double idealpoint = 0.0;
                if (auFrameCounts[aa][kk].getCountSum() != 0) {
                    for (Frame frame : topics[kk].getFrames()) {
                        idealpoint += auFrameCounts[aa][kk].getCount(frame.index)
                                * frame.eta / auFrameCounts[aa][kk].getCountSum();
                    }
                }
                authorFeatures[aa].set(count++, idealpoint);
            }

            for (int kk = 0; kk < K; kk++) {
                for (Frame frame : topics[kk].getFrames()) {
                    double val = 0.0;
                    if (auTopicCounts[aa].getCountSum() != 0) {
                        val = (double) auFrameCounts[aa][kk].getCount(frame.index)
                                / auTopicCounts[aa].getCountSum();
                    }
                    authorFeatures[aa].set(count++, val);
                }
            }

            for (int kk = 0; kk < K; kk++) {
                for (Frame frame : topics[kk].getFrames()) {
                    double val = 0.0;
                    if (auFrameCounts[aa][kk].getCountSum() != 0) {
                        val = (double) auFrameCounts[aa][kk].getCount(frame.index)
                                / auFrameCounts[aa][kk].getCountSum();
                    }
                    authorFeatures[aa].set(count++, val);
                }
            }
            for (int kk = 0; kk < K; kk++) {
                for (Frame frame : topics[kk].getFrames()) {
                    double val = 0.0;
                    if (auFrameCounts[aa][kk].getCountSum() != 0) {
                        val = (double) auFrameCounts[aa][kk].getCount(frame.index)
                                * frame.eta / auFrameCounts[aa][kk].getCountSum();
                    }
                    authorFeatures[aa].set(count++, val);
                }
            }
            authorFeatures[aa].setDimension(count);
        }
        return authorFeatures;
    }

    class Topic {

        final int index;
        DirMult phi;
        final HashMap<Integer, Frame> actives;
        SortedSet<Integer> inactives;
        double[] phihat;
        SparseVector psi;

        public Topic(int idx, DirMult phi) {
            this.index = idx;
            this.phi = phi;
            this.actives = new HashMap<>();
            this.inactives = new TreeSet<Integer>();
            this.psi = new SparseVector();
            this.psi.set(NEW_CHILD_INDEX, 1.0);
        }

        void updatePsi() {
            if (this.psi.size() != getNumFrames() + 1) {
                throw new MismatchRuntimeException(psi.size(), getNumFrames() + 1);
            }

            if (pathAssumption == PathAssumption.MINIMAL) {
                SparseCount counts = new SparseCount();
                for (int dd = 0; dd < D; dd++) {
                    for (int jj : docFramesCounts[dd][index].getIndices()) {
                        counts.increment(jj);
                    }
                }
                this.psi = new SparseVector(getNumFrames() + 1);
                double norm = counts.getCountSum() + frameAlphaGlobal;
                for (int jj : this.getIndices()) {
                    this.psi.set(jj, (double) counts.getCount(jj) / norm);
                }
                this.psi.set(NEW_CHILD_INDEX, (double) frameAlphaGlobal / norm);

            } else if (pathAssumption == PathAssumption.MAXIMAL) {
                SparseCount counts = new SparseCount();
                for (int dd = 0; dd < D; dd++) {
                    for (int jj : docFramesCounts[dd][index].getIndices()) {
                        counts.changeCount(jj, docFramesCounts[dd][index].getCount(jj));
                    }
                }
                this.psi = new SparseVector(getNumFrames() + 1);
                double norm = counts.getCountSum() + frameAlphaGlobal;
                for (int jj : this.getIndices()) {
                    this.psi.set(jj, (double) counts.getCount(jj) / norm);
                }
                this.psi.set(NEW_CHILD_INDEX, (double) frameAlphaGlobal / norm);

            } else if (pathAssumption == PathAssumption.ANTONIAK) {
                SparseCount counts = new SparseCount();
                for (int jj : getIndices()) {
                    for (int dd = 0; dd < D; dd++) {
                        int count = docFramesCounts[dd][index].getCount(jj);
                        if (count > 1) {
                            int c = SamplerUtils.randAntoniak(
                                    frameAlphaLocal * psi.get(jj), count);
                            counts.changeCount(jj, c);
                        } else {
                            counts.changeCount(jj, count);
                        }
                    }
                }

                double[] dirPrior = new double[getNumFrames() + 1];
                ArrayList<Integer> indices = new ArrayList<>();
                int idx = 0;
                for (int jj : getIndices()) {
                    indices.add(jj);
                    dirPrior[idx++] = counts.getCount(jj);
                }
                indices.add(NEW_CHILD_INDEX);
                dirPrior[idx] = frameAlphaGlobal;

                Dirichlet dir = new Dirichlet(dirPrior);
                double[] sampledPsi = dir.nextDistribution();
                this.psi = new SparseVector(sampledPsi.length);
                for (int ii = 0; ii < indices.size(); ii++) {
                    this.psi.set(indices.get(ii), sampledPsi[ii]);
                }
            } else if (pathAssumption == PathAssumption.UNIPROC) {
                this.psi = new SparseVector(getNumFrames() + 1);
                double estProb = 1.0 / (getNumFrames() + frameAlphaGlobal);
                for (int jj : getIndices()) {
                    this.psi.set(jj, estProb);
                }
                this.psi.set(NEW_CHILD_INDEX, estProb * frameAlphaGlobal);
            }
        }

        double[] getPhi() {
            double[] p = new double[V];
            for (int vv = 0; vv < V; vv++) {
                p[vv] = getPhi(vv);
            }
            return p;
        }

        /**
         * Get the probability of a word type given this node. During training,
         * this probability is computed on-the-fly using counts and
         * pseudo-counts. During test, it comes from the learned distribution.
         *
         * @param v word type
         */
        double getPhi(int v) {
            if (this.phihat == null) {
                return phi.getProbability(v);
            }
            return phihat[v];
        }

        public Collection<Frame> getFrames() {
            return this.actives.values();
        }

        public Frame getFrame(int fidx) {
            return this.actives.get(fidx);
        }

        public ArrayList<Integer> getSortedIndices() {
            ArrayList<Integer> sortedIndices = new ArrayList<Integer>();
            for (int ii : getIndices()) {
                sortedIndices.add(ii);
            }
            Collections.sort(sortedIndices);
            return sortedIndices;
        }

        public int getNumFrames() {
            return this.actives.size();
        }

        public Set<Integer> getIndices() {
            return this.actives.keySet();
        }

        public boolean isEmpty() {
            return actives.isEmpty();
        }

        public boolean isActive(int idx) {
            return this.actives.containsKey(idx);
        }

        public void removeComponent(int idx) {
            this.inactives.add(idx);
            this.actives.remove(idx);
        }

        public void createNewComponent(int idx, Frame c) {
            if (isActive(idx)) {
                throw new RuntimeException("Component " + idx + " exists");
            }
            if (inactives.contains(idx)) {
                inactives.remove(idx);
            }
            this.actives.put(idx, c);
        }

        public void fillInactives() {
            int maxTableIndex = -1;
            for (int idx : actives.keySet()) {
                if (idx > maxTableIndex) {
                    maxTableIndex = idx;
                }
            }
            this.inactives = new TreeSet<Integer>();
            for (int ii = 0; ii < maxTableIndex; ii++) {
                if (!isActive(ii)) {
                    inactives.add(ii);
                }
            }
        }

        public int getNextIndex() {
            int newIdx;
            if (this.inactives.isEmpty()) {
                newIdx = this.actives.size();
            } else {
                newIdx = this.inactives.first();
            }
            return newIdx;
        }
    }

    class Frame {

        final int index;
        final int topicIndex;
        final int born;
        DirMult phi;
        double eta;
        boolean isNew;
        double[] phihat;
        ArrayList<Double> etaList; // store eta over time

        public Frame(int idx, int topicIdx, int born, DirMult phi, double eta) {
            this.index = idx;
            this.topicIndex = topicIdx;
            this.born = born;
            this.phi = phi;
            this.eta = eta;
            this.isNew = true;
            this.etaList = new ArrayList<>();
        }

        /**
         * Store the current eta.
         */
        public void storeEta() {
            this.etaList.add(eta);
        }

        public void changeToNew() {
            this.isNew = true;
        }

        public void changeToNormal() {
            this.isNew = false;
        }

        double[] getPhi() {
            double[] p = new double[V];
            for (int vv = 0; vv < V; vv++) {
                p[vv] = getPhi(vv);
            }
            return p;
        }

        /**
         * Get the probability of a word type given this node. During training,
         * this probability is computed on-the-fly using counts and
         * pseudo-counts. During test, it comes from the learned distribution.
         *
         * @param v word type
         */
        double getPhi(int v) {
            if (this.phihat == null) {
                return phi.getProbability(v);
            }
            return phihat[v];
        }
    }

    /**
     * Sample topic assignments for test documents and make predictions using
     * the final model.
     */
    public SparseVector[] test() {
        return this.test(null, null, null);
    }

    /**
     * Sample topic assignments for test documents and make predictions.
     *
     * @param stateFile
     * @param predictionFile
     * @param testReportFolder
     * @return predictions
     */
    public SparseVector[] test(File stateFile,
            File predictionFile,
            File testReportFolder) {
        if (authorIndices == null) {
            throw new RuntimeException("List of test authors is null");
        }

        if (stateFile == null) {
            stateFile = getFinalStateFile();
        }

        setTestConfigurations(); // set up test
        if (verbose) {
            logln("Setting up test ...");
            logln("--- test burn-in: " + testBurnIn);
            logln("--- test maximum number of iterations: " + testMaxIter);
            logln("--- test sample lag: " + testSampleLag);
            logln("--- test report interval: " + testRepInterval);
        }

        // sample on test data
        ArrayList<SparseVector[]> predictionList = sampleNewDocuments(stateFile, testReportFolder);

        // make prediction on votes of unknown voters
        SparseVector[] predictions = averagePredictions(predictionList);

        if (predictionFile != null) { // output predictions
            AbstractVotePredictor.outputPredictions(predictionFile, null, predictions);
        }
        return predictions;
    }

    public double getTestVoteLogLikelihood() {
        double llh = 0.0;
        int count = 0;
        for (int aa = 0; aa < A; aa++) {
            for (int bb = 0; bb < B; bb++) {
                if (isValidVote(aa, bb)) {
                    double score = u[aa] * x[bb] + y[bb];
                    llh += votes[aa][bb] * score - Math.log(1 + Math.exp(score));
                    count++;
                }
            }
        }
        return llh / count;
    }

    /**
     * Sample topic assignments for all tokens in a set of test documents.
     *
     * @param stateFile
     * @param testWords
     * @param testDocIndices
     * @param testReportFolder
     */
    private ArrayList<SparseVector[]> sampleNewDocuments(
            File stateFile,
            File testReportFolder) {
        if (verbose) {
            logln("--- Sampling on test data using " + stateFile);
        }
        if (stateFile == null) {
            throw new RuntimeException("Specifying stateFile. " + stateFile);
        }

        if (testReportFolder == null) {
            testReportFolder = new File(this.getSamplerFolderPath(), "te_" + ReportFolder);
        }
        IOUtils.createFolder(testReportFolder);

        inputBillIdealPoints(stateFile.getAbsolutePath());
        initializeDataStructure();
        initializeBillThetas();

        inputModel(stateFile.getAbsolutePath());
        // remove training data assigned
        for (int kk = 0; kk < K; kk++) {
            topics[kk].phi.clear();
            for (Frame frame : topics[kk].getFrames()) {
                frame.phi.clear();
            }
        }

        if (votes != null) {
            u = new double[A];
            for (int aa = 0; aa < A; aa++) {
                u[aa] = SamplerUtils.getGaussian(0.0, sigma);
            }

            int maxII = 1000;
            int step = MiscUtils.getRoundStepSize(maxII, 10);
            for (int ii = 0; ii < maxII; ii++) {
                rate = getAnnealingRate(epsilon / 10, ii, maxII);
                for (int a = 0; a < A; a++) {
                    if (!validAs[a]) {
                        continue;
                    }
                    double llh = 0.0;
                    for (int b = 0; b < votes[a].length; b++) {
                        if (isValidVote(a, b)) {
                            double score = Math.exp(u[a] * x[b] + y[b]);
                            double prob = score / (1 + score);
                            llh += x[b] * (votes[a][b] - prob); // only work for 0 and 1
                        }
                    }
                    u[a] += (llh - (u[a]) / sigma) * rate / B;
                }

                if (ii % step == 0) {
                    logln("--- vote llh: " + getTestVoteLogLikelihood());
                }
            }
        }

        // sample
        ArrayList<SparseVector[]> predictionList = new ArrayList<>();
        for (iter = 0; iter < testMaxIter; iter++) {
            isReporting = verbose && iter % testRepInterval == 0;
            if (isReporting) {
                String str = "Iter " + iter + "/" + testMaxIter;
                if (iter < testBurnIn) {
                    logln("--- Burning in. " + str);
                } else {
                    logln("--- Sampling. " + str);
                }
            }

            // sample topic assignments
            if (iter == 0) {
                sampleZs(!REMOVE, !ADD, !REMOVE, ADD, !OBSERVED, !EXTEND);
            } else {
                sampleZs(!REMOVE, !ADD, REMOVE, ADD, !OBSERVED, !EXTEND);
            }

            // update us
            us = new double[A][K];
            for (int aa = 0; aa < A; aa++) {
                for (int kk = 0; kk < K; kk++) {
                    if (!isBackedOff(aa, kk)) {
                        us[aa][kk] = getLexicalU(aa, kk);
                    }
                }
            }

            if (iter % testSampleLag == 0) {
                String filename = IOUtils.removeExtension(IOUtils.getFilename(stateFile.getAbsolutePath()));
                outputState(new File(testReportFolder, filename + ".zip"));

//                SparseVector[] predictions = predictOutMatrix();
//                if (iter >= testBurnIn) { // store partial prediction
//                    predictionList.add(predictions);
//                }
//
//                if (votes != null) { // for debug
//                    ArrayList<Measurement> measurements = AbstractVotePredictor
//                            .evaluate(votes, validVotes, predictions);
//                    for (Measurement m : measurements) {
//                        logln(">>> o >>> " + m.getName() + ": " + m.getValue());
//                    }
//                }
            }
        }

        try {

        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while outputing to " + testReportFolder);
        }
        return predictionList;
    }

    /**
     * Run Gibbs sampling on test data using multiple models learned which are
     * stored in the ReportFolder. The runs on multiple models are parallel.
     *
     * @param newWords Words of new documents
     * @param newDocIndices Indices of selected documents
     * @param newAuthors
     * @param newAuthorIndices
     * @param testVotes
     * @param iterPredFolder Output folder
     * @param sampler The configured sampler
     * @return Predicted probabilities
     */
    public static SparseVector[] parallelTest(ArrayList<Integer> newDocIndices,
            int[][] newWords,
            int[] newAuthors,
            int[][] votes,
            ArrayList<Integer> newAuthorIndices,
            boolean[][] testVotes,
            File iterPredFolder, File testReportFolder,
            HierMultSHDP sampler) {
        File reportFolder = new File(sampler.getSamplerFolderPath(), ReportFolder);
        if (!reportFolder.exists()) {
            throw new RuntimeException("Report folder not found. " + reportFolder);
        }
        SparseVector[] predictions = null;
        String[] filenames = reportFolder.list();
        try {
            IOUtils.createFolder(iterPredFolder);
            ArrayList<Thread> threads = new ArrayList<Thread>();
            ArrayList<File> partPredFiles = new ArrayList<>();
            for (String filename : filenames) {
                if (!filename.contains("zip")) {
                    continue;
                }

                File stateFile = new File(reportFolder, filename);
                File partialResultFile = new File(iterPredFolder,
                        IOUtils.removeExtension(filename) + ".txt");
                HierMultSHDPRunner runner = new HierMultSHDPRunner(
                        sampler, stateFile, newDocIndices, newWords,
                        newAuthors, votes, newAuthorIndices, testVotes,
                        partialResultFile, testReportFolder);
                Thread thread = new Thread(runner);
                threads.add(thread);
                partPredFiles.add(partialResultFile);
            }

            // run MAX_NUM_PARALLEL_THREADS threads at a time
            runThreads(threads);

            // average over multiple predictions
            for (File partPredFile : partPredFiles) {
                SparseVector[] partPredictions = AbstractVotePredictor.inputPredictions(partPredFile);
                if (predictions == null) {
                    predictions = partPredictions;
                } else {
                    if (predictions.length != partPredictions.length) {
                        throw new RuntimeException("Mismatch");
                    }
                    for (int aa : newAuthorIndices) {
                        predictions[aa].add(partPredictions[aa]);
                    }
                }
            }
            for (int aa : newAuthorIndices) {
                predictions[aa].scale(1.0 / partPredFiles.size());
            }
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while sampling during parallel test.");
        }
        return predictions;
    }
}

class HierMultSHDPRunner implements Runnable {

    HierMultSHDP sampler;
    File stateFile;
    ArrayList<Integer> testDocIndices;
    int[][] testWords;
    int[] testAuthors;
    int[][] votes;
    ArrayList<Integer> testAuthorIndices;
    boolean[][] testVotes;
    File predictionFile;
    File testReportFolder;

    public HierMultSHDPRunner(HierMultSHDP sampler,
            File stateFile,
            ArrayList<Integer> newDocIndices,
            int[][] newWords,
            int[] newAuthors,
            int[][] votes,
            ArrayList<Integer> newAuthorIndices,
            boolean[][] testVotes,
            File outputFile,
            File testReportFolder) {
        this.sampler = sampler;
        this.stateFile = stateFile;
        this.testDocIndices = newDocIndices;
        this.testWords = newWords;
        this.testAuthors = newAuthors;
        this.testAuthorIndices = newAuthorIndices;
        this.testVotes = testVotes;
        this.predictionFile = outputFile;
        this.testReportFolder = testReportFolder;
    }

    @Override
    public void run() {
        HierMultSHDP testSampler = new HierMultSHDP();
        testSampler.setVerbose(true);
        testSampler.setDebug(false);
        testSampler.setLog(false);
        testSampler.setReport(false);
        testSampler.configure(sampler);
        testSampler.setWordVocab(sampler.getWordVocab());
        testSampler.setTopicVocab(sampler.getTopicVocab());
        testSampler.setTestConfigurations(sampler.getBurnIn(),
                sampler.getMaxIters(), sampler.getSampleLag());
        testSampler.setupData(testDocIndices, testWords, testAuthors, votes,
                testAuthorIndices, null, testVotes);
        testSampler.setBillWords(sampler.billWords);
        try {
            testSampler.test(stateFile, predictionFile, testReportFolder);
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException();
        }
    }
}
