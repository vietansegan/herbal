package votepredictor.textidealpoint;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import util.IOUtils;
import util.MiscUtils;
import util.SparseVector;

/**
 *
 * @author vietan
 */
public class RecursiveSLDAIdealPoint extends AbstractTextSingleIdealPoint {

    public static final int ALPHA = 0;
    public static final int BETA = 1;
    public static final int INVALID_TOPIC = -1;
    public double mu;       // eta mean
    public double gamma;    // eta variance
    // input
    protected int K; // number of topics
    protected int L = 2;
    protected double[][] priorTopics;

    protected int[][][] zs;
    protected AuthorRSLDA root;

    protected ArrayList<String> topicVocab;

    public RecursiveSLDAIdealPoint() {
        this.basename = "Recursive-SLDA-ideal-point";
    }

    public RecursiveSLDAIdealPoint(String bname) {
        this.basename = bname;
    }

    public void setTopicVocab(ArrayList<String> topicVoc) {
        this.topicVocab = topicVoc;
    }

    public void setPriorTopics(double[][] priors) {
        this.priorTopics = priors;
    }

    public void configure(
            String folder,
            int V, int K,
            double alpha,
            double beta,
            double rho,
            double sigma,
            double mu,
            double gamma,
            InitialState initState,
            boolean paramOpt,
            int burnin, int maxiter, int samplelag, int repInt) {
        if (verbose) {
            logln("Configuring ...");
        }

        this.folder = folder;
        this.K = K;
        this.V = V;

        this.hyperparams = new ArrayList<Double>();
        this.hyperparams.add(alpha);
        this.hyperparams.add(beta);
        this.rho = rho;
        this.mu = mu;
        this.sigma = sigma;
        this.gamma = gamma;
        this.wordWeightType = WordWeightType.NONE;

        this.sampledParams = new ArrayList<ArrayList<Double>>();
        this.sampledParams.add(cloneHyperparameters());

        this.BURN_IN = burnin;
        this.MAX_ITER = maxiter;
        this.LAG = samplelag;
        this.REP_INTERVAL = repInt;

        this.initState = initState;
        this.paramOptimized = paramOpt;
        this.prefix += initState.toString();
        this.setName();

        this.report = true;

        if (verbose) {
            logln("--- folder\t" + folder);
            logln("--- num topics:\t" + K);
            logln("--- num word types:\t" + V);
            logln("--- alpha:\t" + MiscUtils.formatDouble(hyperparams.get(ALPHA)));
            logln("--- beta:\t" + MiscUtils.formatDouble(hyperparams.get(BETA)));
            logln("--- rho:\t" + MiscUtils.formatDouble(this.rho));
            logln("--- sigma:\t" + MiscUtils.formatDouble(this.sigma));
            logln("--- reg mu:\t" + MiscUtils.formatDouble(this.mu));
            logln("--- reg gamma:\t" + MiscUtils.formatDouble(this.gamma));
            logln("--- burn-in:\t" + BURN_IN);
            logln("--- max iter:\t" + MAX_ITER);
            logln("--- sample lag:\t" + LAG);
            logln("--- paramopt:\t" + paramOptimized);
            logln("--- initialize:\t" + initState);
        }
    }

    protected void setName() {
        StringBuilder str = new StringBuilder();
        str.append(this.prefix)
                .append("_").append(basename)
                .append("_B-").append(BURN_IN)
                .append("_M-").append(MAX_ITER)
                .append("_L-").append(LAG)
                .append("_K-").append(K)
                .append("_a-").append(formatter.format(hyperparams.get(ALPHA)))
                .append("_b-").append(formatter.format(hyperparams.get(BETA)))
                .append("_r-").append(formatter.format(rho))
                .append("_s-").append(formatter.format(sigma))
                .append("_m-").append(formatter.format(mu))
                .append("_g-").append(formatter.format(gamma));
        str.append("_opt-").append(this.paramOptimized);
        this.name = str.toString();
    }

    /**
     * Make predictions on held-out votes of unknown legislators on known votes.
     *
     * @return Predicted probabilities
     */
    @Override
    public SparseVector[] predictOutMatrix() {
        SparseVector[] predictions = new SparseVector[validVotes.length];
        return predictions;
    }

    public int[][][] getAssingments() {
        return this.zs;
    }

    @Override
    public void initialize() {
        if (verbose) {
            logln("Initializing ...");
        }

        initializeIdealPoint();

        zs = new int[L][][];
        for (int l = 0; l < L; l++) {
            zs[l] = new int[D][];
            for (int d = 0; d < D; d++) {
                zs[l][d] = new int[words[d].length];
            }
        }

        if (debug) {
            validate("Initialized");
        }
    }

    @Override
    public void iterate() {
        File recursiveFolder = new File(this.getSamplerFolderPath(), "recursive");
        IOUtils.createFolder(recursiveFolder);

        boolean[][] validTokens = new boolean[D][];
        for (int d = 0; d < D; d++) {
            validTokens[d] = new boolean[words[d].length];
            Arrays.fill(validTokens[d], true);
        }

        root = new AuthorRSLDA();
        root.setVerbose(verbose);
        root.setDebug(debug);
        root.setLog(true);
        root.setReport(true);
        root.setWordVocab(wordVocab);
        File rootFolder = new File(recursiveFolder, "root");
        IOUtils.createFolder(rootFolder);
        root.configure(rootFolder.getAbsolutePath(),
                V, K, A,
                hyperparams.get(ALPHA), hyperparams.get(BETA),
                rho, mu, gamma,
                initState, paramOptimized,
                BURN_IN, MAX_ITER, LAG, REP_INTERVAL);
        root.setupData(words, validTokens, authors, u);

        File rootFile = root.getFinalStateFile();
        if (rootFile.exists()) {
            root.inputFinalState();
        } else {
            root.initialize(priorTopics);
            root.metaIterate();
        }
        for (int d = 0; d < D; d++) {
            for (int n = 0; n < words[d].length; n++) {
                if (root.getValidToken(d, n)) {
                    zs[0][d][n] = root.z[d][n];
                }
            }
        }
        root.outputTopicTopWords(new File(root.getSamplerFolderPath(),
                TopWordFile), 20, topicVocab);

        // Run for each issue
        for (int kk = 0; kk < K; kk++) {
            AuthorRSLDA slda = new AuthorRSLDA();
            slda.setVerbose(verbose);
            slda.setDebug(debug);
            slda.setLog(true);
            slda.setReport(true);
            slda.setWordVocab(wordVocab);
            File samplerFolder = new File(recursiveFolder, "slda-" + kk);
            IOUtils.createFolder(samplerFolder);
            slda.configure(samplerFolder.getAbsolutePath(),
                    V, 5, A,
                    hyperparams.get(ALPHA), hyperparams.get(BETA),
                    rho, mu, gamma,
                    initState, paramOptimized,
                    BURN_IN, MAX_ITER, LAG, REP_INTERVAL);

            boolean[][] subValidTokens = new boolean[D][];
            for (int d = 0; d < D; d++) {
                subValidTokens[d] = new boolean[words[d].length];
                Arrays.fill(subValidTokens[d], false);
                for (int n = 0; n < words[d].length; n++) {
                    if (zs[0][d][n] == kk) {
                        subValidTokens[d][n] = true;
                    }
                }
            }

            slda.setupData(words, subValidTokens, authors, u);
            slda.initialize();
            slda.metaIterate();
            slda.outputTopicTopWords(new File(slda.getSamplerFolderPath(),
                    TopWordFile), 20);
        }
    }

    @Override
    protected void updateUs() {

    }

    @Override
    public double getLogLikelihood() {
        return 0;
    }

    @Override
    public double getLogLikelihood(ArrayList<Double> newParams) {
        return 0;
    }

    @Override
    public void updateHyperparameters(ArrayList<Double> newParams) {
    }

    @Override
    public void validate(String msg) {
    }

    @Override
    public void outputState(String filepath) {

    }

    @Override
    public void inputState(String filepath) {

    }
}
