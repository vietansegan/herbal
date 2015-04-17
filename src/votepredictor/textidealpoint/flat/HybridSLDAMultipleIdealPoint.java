package votepredictor.textidealpoint.flat;

import edu.stanford.nlp.optimization.DiffFunction;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import optimization.OWLQN;
import sampler.unsupervised.LDA;
import sampling.likelihood.DirMult;
import util.IOUtils;
import util.MiscUtils;
import util.MismatchRuntimeException;
import util.RankingItem;
import util.SamplerUtils;
import util.SparseVector;
import util.StatUtils;
import util.evaluation.Measurement;
import votepredictor.AbstractVotePredictor;
import votepredictor.textidealpoint.AbstractTextMultipleIdealPoint;

/**
 *
 * @author vietan
 */
public class HybridSLDAMultipleIdealPoint extends AbstractTextMultipleIdealPoint {

    public static final int ALPHA = 0;
    public static final int BETA = 1;
    public double gamma; // eta variance
    public double l1; // l1-norm using OWL-QN
    public double l2; // l2-norm using OWL-QN

    // input
    protected double[][] priorTopics;

    // latent variables
    protected DirMult[] topicWords;
    protected DirMult[] docTopics;
    protected int[][] z;

    protected double[] eta; // regression parameters for topics
//    protected SparseVector[] tau; // each topic has its own lexical regression
    protected SparseVector[] topVals;
//    protected SparseVector[] lexVals;

    private ArrayList<String> labelVocab;
    protected boolean isLexReg;
    protected boolean isTopReg;

    private double sqrtRho;

    public HybridSLDAMultipleIdealPoint() {
        this.basename = "Hybrid-SLDA-mult-ideal-point";
    }

    public HybridSLDAMultipleIdealPoint(String bname) {
        this.basename = bname;
    }

    public void setLabelVocab(ArrayList<String> labelVoc) {
        this.labelVocab = labelVoc;
    }

    public void configure(
            String folder,
            int V, int K,
            double alpha,
            double beta,
            double rho,
            double sigma,
            double gamma,
            double l1, double l2,
            double[][] priorTopics,
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
        this.sqrtRho = Math.sqrt(this.rho);
        this.sigma = sigma;
        this.gamma = gamma;
        this.l1 = l1;
        this.l2 = l2;
        this.priorTopics = priorTopics;
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
        this.report = true;

        this.isTopReg = this.K > 0;
        this.isLexReg = this.l1 > 0 || this.l2 > 0;

        StringBuilder str = new StringBuilder();
        str.append(this.prefix)
                .append("_").append(basename)
                .append(isTopReg ? ("_B-" + BURN_IN) : "")
                .append(isTopReg ? ("_M-" + MAX_ITER) : "")
                .append(isTopReg ? ("_L-" + LAG) : "")
                .append(isTopReg ? ("_K-" + K) : "")
                .append("_r-").append(formatter.format(rho))
                .append("_s-").append(formatter.format(sigma)) // ideal point var
                .append(isTopReg ? ("_a-" + formatter.format(hyperparams.get(ALPHA))) : "")
                .append(isTopReg ? ("_b-" + formatter.format(hyperparams.get(BETA))) : "")
                .append(isTopReg ? ("_g-" + formatter.format(gamma)) : "")
                .append("_l1-").append(MiscUtils.formatDouble(l1, 10))
                .append("_l2-").append(MiscUtils.formatDouble(l2, 10));
        str.append("_opt-").append(this.paramOptimized);
        str.append("_prior-").append(this.priorTopics != null);
        this.name = str.toString();

        if (verbose) {
            logln("--- folder\t" + folder);
            logln("--- num topics:\t" + K);
            logln("--- num word types:\t" + V);
            logln("--- alpha:\t" + MiscUtils.formatDouble(hyperparams.get(ALPHA)));
            logln("--- beta:\t" + MiscUtils.formatDouble(hyperparams.get(BETA)));
            logln("--- rho:\t" + MiscUtils.formatDouble(rho));
            logln("--- sigma:\t" + MiscUtils.formatDouble(sigma));
            logln("--- gamma:\t" + MiscUtils.formatDouble(gamma));
            logln("--- l1:\t" + MiscUtils.formatDouble(l1, 10));
            logln("--- l2:\t" + MiscUtils.formatDouble(l2, 10));
            logln("--- burn-in:\t" + BURN_IN);
            logln("--- max iter:\t" + MAX_ITER);
            logln("--- sample lag:\t" + LAG);
            logln("--- paramopt:\t" + paramOptimized);
            logln("--- initialize:\t" + initState);
            logln("--- isLexReg? " + isLexReg);
            logln("--- isTopReg? " + isTopReg);
            logln("--- has prior? " + (priorTopics != null));
        }
    }

    @Override
    public void initialize() {
        initialize(priorTopics);
    }

    public void initialize(double[][] seededTopics) {
        if (verbose) {
            logln("Initializing ...");
        }
        iter = INIT;
        isReporting = true;

        initializeIdealPoint(); // initialize ideal points
        initializeModelStructure(seededTopics);
        initializeDataStructure();
        initializeAssignments();

        if (debug) {
            validate("Initialized");
        }

        if (verbose) {
            logln("--- Done initializing. \n" + getCurrentState());
            getLogLikelihood();
        }
    }

    protected void initializeIdealPoint() {
        us = new double[A][K];
        for (int aa = 0; aa < A; aa++) {
            if (validAs[aa]) {
                for (int kk = 0; kk < K; kk++) {
                    us[aa][kk] = SamplerUtils.getGaussian(0.0, 3.0);
                }
            }
        }

        xs = new double[B][K + 1];
        for (int bb = 0; bb < B; bb++) {
            if (validBs[bb]) {
                for (int kk = 0; kk < K + 1; kk++) {
                    xs[bb][kk] = SamplerUtils.getGaussian(0.0, 3.0);
                }
            }
        }
    }

    protected void initializeModelStructure(double[][] seededTopics) {
        if (seededTopics != null && seededTopics.length != K) {
            throw new RuntimeException("Mismatch" + ". K = " + K
                    + ". # prior topics = " + seededTopics.length);
        }

        topicWords = new DirMult[K];
        for (int k = 0; k < K; k++) {
            if (seededTopics != null) {
                topicWords[k] = new DirMult(V, hyperparams.get(BETA) * V, seededTopics[k]);
            } else {
                topicWords[k] = new DirMult(V, hyperparams.get(BETA) * V, 1.0 / V);
            }
        }

        eta = new double[K];
//        tau = new SparseVector[K];
//        for (int kk = 0; kk < K; kk++) {
//            tau[kk] = new SparseVector(V);
//        }
    }

    protected void initializeDataStructure() {
        z = new int[D][];
        for (int d = 0; d < D; d++) {
            z[d] = new int[words[d].length];
        }

        docTopics = new DirMult[D];
        for (int d = 0; d < D; d++) {
            docTopics[d] = new DirMult(K, hyperparams.get(ALPHA) * K, 1.0 / K);
        }
        topVals = new SparseVector[A];
//        lexVals = new SparseVector[A];
        for (int aa = 0; aa < A; aa++) {
            topVals[aa] = new SparseVector(K);
//            lexVals[aa] = new SparseVector(K);
        }
    }

    protected void initializeAssignments() {
        switch (initState) {
            case RANDOM:
                initializeRandomAssignments();
                break;
            case PRESET:
                initializePresetAssignments();
                break;
            default:
                throw new RuntimeException("Initialization " + initState + " not supported");
        }
    }

    protected void initializeRandomAssignments() {
        if (verbose) {
            logln("--- Initializing random assignments ...");
        }

        sampleZs(!REMOVE, ADD, !REMOVE, ADD, !OBSERVED);
    }

    protected void initializePresetAssignments() {
        if (verbose) {
            logln("--- Initializing preset assignments. Running LDA ...");
        }

        LDA lda = this.runLDA(K);
        int[][] ldaZ = lda.getZs();

        // initialize assignments
        for (int dd = 0; dd < D; dd++) {
            int aa = authors[dd];
            for (int nn = 0; nn < words[dd].length; nn++) {
                z[dd][nn] = ldaZ[dd][nn];
                docTopics[dd].increment(z[dd][nn]);
                topicWords[z[dd][nn]].increment(words[dd][nn]);
                topVals[aa].change(z[dd][nn], eta[z[dd][nn]] / authorTotalWordWeights[aa]);
//                lexVals[aa].change(z[dd][nn], tau[z[dd][nn]].get(words[dd][nn]) / authorTotalWordWeights[aa]);
            }
        }
    }

    public SparseVector[] predictOutMatrix() {
        SparseVector[] predictions = new SparseVector[validVotes.length];
        for (int aa = 0; aa < A; aa++) {
            int author = authorIndices.get(aa);
            predictions[author] = new SparseVector(validVotes[author].length);
            for (int bb = 0; bb < B; bb++) {
                int bill = billIndices.get(bb);
                if (isValidVote(aa, bb)) {
                    double dotprod = xs[bb][K];
                    for (int kk = 0; kk < K; kk++) {
//                        dotprod += (topVals[aa].get(kk) + lexVals[aa].get(kk)) * xs[bb][kk];
                        dotprod += topVals[aa].get(kk) * xs[bb][kk];
                    }
                    double score = Math.exp(dotprod);
                    double prob = score / (1.0 + score);
                    predictions[author].set(bill, prob);
                }
            }
        }
        return predictions;
    }

    protected double getMSE() {
        double mse = 0.0;
        for (int aa = 0; aa < A; aa++) {
            for (int kk = 0; kk < K; kk++) {
//                double diff = us[aa][kk] - (topVals[aa].get(kk) + lexVals[aa].get(kk));
                double diff = us[aa][kk] - topVals[aa].get(kk);
                mse += diff * diff;
            }
        }
        return mse / (A * K);
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
        logln("--- MSE: " + getMSE());
        return str;
    }

    @Override
    public void iterate() {
//        updateTaus();
        updateEtas();
        sampleZs(REMOVE, ADD, REMOVE, ADD, OBSERVED);
        updateUX();
    }

    protected long updateUX() {
        if (isReporting) {
            logln("+++ Updating U & X ...");
        }
        long sTime = System.currentTimeMillis();

        for (int ii = 0; ii < 10; ii++) {
            updateUs();
            updateXs();
        }

        long eTime = System.currentTimeMillis() - sTime;
        if (isReporting) {
            logln("--- --- time: " + eTime);
        }
        return eTime;
    }

    protected long updateXs() {
        long sTime = System.currentTimeMillis();
        double bRate = 0.01;
        for (int bb = 0; bb < B; bb++) {
            if (!validBs[bb]) {
                continue;
            }
            double[] gradXs = new double[K + 1];
            for (int aa = 0; aa < A; aa++) {
                if (isValidVote(aa, bb)) {
                    double dotprod = xs[bb][K];
                    for (int kk = 0; kk < K; kk++) {
                        dotprod += us[aa][kk] * xs[bb][kk];
                    }
                    double score = Math.exp(dotprod);
                    double prob = score / (1 + score);
                    for (int kk = 0; kk < K; kk++) {
                        gradXs[kk] += us[aa][kk] * (getVote(aa, bb) - prob);
                    }
                    gradXs[K] += getVote(aa, bb) - prob;
                }
            }
            for (int kk = 0; kk < K + 1; kk++) {
                gradXs[kk] -= xs[bb][kk] / sigma;
                xs[bb][kk] += bRate * gradXs[kk];
            }
        }
        long eTime = System.currentTimeMillis() - sTime;
        return eTime;
    }

    private long updateUs() {
        long sTime = System.currentTimeMillis();
        double aRate = 0.01;
        for (int aa = 0; aa < A; aa++) {
            if (!validAs[aa]) {
                continue;
            }
            double[] grads = new double[K];
            for (int bb = 0; bb < B; bb++) {
                if (isValidVote(aa, bb)) {
                    double dotprod = xs[bb][K];
                    for (int kk = 0; kk < K; kk++) {
                        dotprod += us[aa][kk] * xs[bb][kk];
                    }
                    double score = Math.exp(dotprod);
                    double prob = score / (1 + score);
                    for (int kk = 0; kk < K; kk++) {
                        grads[kk] += xs[bb][kk] * (getVote(aa, bb) - prob);
                    }
                }
            }
            for (int kk = 0; kk < K; kk++) {
//                grads[kk] -= (us[aa][kk] - topVals[aa].get(kk) - lexVals[aa].get(kk)) / rho;
                grads[kk] -= (us[aa][kk] - topVals[aa].get(kk)) / rho;
                us[aa][kk] += aRate * grads[kk];
            }
        }
        long eTime = System.currentTimeMillis() - sTime;
        return eTime;
    }

    /**
     * Sample topic assignment for each token.
     *
     * @param removeFromModel
     * @param addToModel
     * @param removeFromData
     * @param addToData
     * @param observe
     * @return Elapsed time
     */
    protected long sampleZs(
            boolean removeFromModel, boolean addToModel,
            boolean removeFromData, boolean addToData,
            boolean observe) {
        if (isReporting) {
            logln("+++ Sampling assignments ...");
        }

        long sTime = System.currentTimeMillis();
        numTokensChanged = 0;
        for (int d = 0; d < D; d++) {
            int aa = authors[d];
            for (int n = 0; n < words[d].length; n++) {
                if (removeFromModel) {
                    topicWords[z[d][n]].decrement(words[d][n]);
                }
                if (removeFromData) {
                    docTopics[d].decrement(z[d][n]);
                    topVals[aa].change(z[d][n], -eta[z[d][n]] / authorTotalWordWeights[aa]);
//                    lexVals[aa].change(z[d][n], -tau[z[d][n]].get(words[d][n]) / authorTotalWordWeights[aa]);
                }

                // compute current llhs
                double[] uLlhs = new double[K];
                if (observe) {
                    for (int kk = 0; kk < K; kk++) {
//                        double mean = topVals[aa].get(kk) + lexVals[aa].get(kk);
                        double mean = topVals[aa].get(kk);
                        uLlhs[kk] = StatUtils.logNormalProbability(us[aa][kk], mean, sqrtRho);
                    }
                }

                double[] logprobs = new double[K];
                for (int kk = 0; kk < K; kk++) {
                    logprobs[kk]
                            = Math.log(docTopics[d].getCount(kk) + hyperparams.get(ALPHA))
                            + Math.log(topicWords[kk].getProbability(words[d][n]));
                    if (observe) {
//                        double mean = topVals[aa].get(kk) + lexVals[aa].get(kk)
//                              + (eta[kk] + tau[kk].get(words[d][n])) / authorTotalWordWeights[aa];
                        double mean = topVals[aa].get(kk)
                                + eta[kk] / authorTotalWordWeights[aa];
                        double resLlh = StatUtils.logNormalProbability(us[aa][kk], mean, sqrtRho);
                        logprobs[kk] += resLlh - uLlhs[kk];
                    }
                }
                int sampledZ = SamplerUtils.logMaxRescaleSample(logprobs);
                if (sampledZ == K) {
                    logln("iter = " + iter + ". d = " + d + ". n = " + n);
                    for (int kk = 0; kk < K; kk++) {
                        logln("k = " + kk
                                + ". " + (Math.log(docTopics[d].getCount(kk) + hyperparams.get(ALPHA)))
                                + ". " + (Math.log(topicWords[kk].getProbability(words[d][n]))));
                    }
                    throw new RuntimeException("Out-of-bound sample. "
                            + "SampledZ = " + sampledZ);
                }

                if (z[d][n] != sampledZ) {
                    numTokensChanged++; // for debugging
                }
                // update
                z[d][n] = sampledZ;

                if (addToModel) {
                    topicWords[z[d][n]].increment(words[d][n]);
                }
                if (addToData) {
                    docTopics[d].increment(z[d][n]);
                    topVals[aa].change(z[d][n], eta[z[d][n]] / authorTotalWordWeights[aa]);
//                    lexVals[aa].change(z[d][n], tau[z[d][n]].get(words[d][n]) / authorTotalWordWeights[aa]);
                }
            }
        }
        long eTime = System.currentTimeMillis() - sTime;
        if (isReporting) {
            logln("--- --- time: " + eTime);
            logln("--- --- # tokens: " + numTokens
                    + ". # tokens changed: " + numTokensChanged
                    + " (" + MiscUtils.formatDouble((double) numTokensChanged / numTokens) + ")");
        }
        return eTime;
    }

    /**
     * Optimize etas using L-BFGS.
     *
     * @return Elapsed time
     */
    public long updateEtas() {
        if (isReporting) {
            logln("+++ Updating etas ...");
        }
        long sTime = System.currentTimeMillis();

        OWLQN minimizer = new OWLQN();
        minimizer.setQuiet(true);
        minimizer.setMaxIters(100);

        EtaDiffFunc diff = new EtaDiffFunc();
        minimizer.minimize(diff, eta, 0.0);

        long eTime = System.currentTimeMillis() - sTime;
        if (isReporting) {
            logln("--- --- time: " + eTime);
        }
        return eTime;
    }

//    public long updateTaus() {
//        if (isReporting) {
//            logln("+++ Updating taus ...");
//        }
//        long sTime = System.currentTimeMillis();
//
//        for (int kk = 0; kk < K; kk++) {
//            OWLQN minimizer = new OWLQN();
//            minimizer.setQuiet(true);
//            minimizer.setMaxIters(100);
//
//            TauDiffFunc diff = new TauDiffFunc(kk);
//            double[] topicTau = tau[kk].dense();
//            try {
//                minimizer.minimize(diff, topicTau, l1);
//            } catch (Exception e) {
//                e.printStackTrace();
//                System.out.println("Not converged. iter = " + iter + ". kk = " + kk);
//            }
//            tau[kk] = new SparseVector(topicTau);
//        }
//
//        long eTime = System.currentTimeMillis() - sTime;
//        if (isReporting) {
//            logln("--- --- time: " + eTime);
//        }
//        return eTime;
//    }
    
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

    }

    @Override
    public void outputState(String filepath) {
        if (verbose) {
            logln("--- Outputing current state to " + filepath);
        }
        try {
            // model
            StringBuilder modelStr = new StringBuilder();
            for (int vv = 0; vv < V; vv++) {
                modelStr.append(vv).append("\t").append(this.wordWeights[vv]).append("\n");
            }
            for (int kk = 0; kk < K; kk++) {
                modelStr.append(kk).append("\n");
                modelStr.append(eta[kk]).append("\n");
                modelStr.append(DirMult.output(topicWords[kk])).append("\n");
            }
//            for (int kk = 0; kk < K; kk++) {
//                modelStr.append(SparseVector.output(tau[kk])).append("\n");
//            }

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
                billStr.append(MiscUtils.arrayToString(xs[bb])).append("\n");
            }

            // assignments
            StringBuilder assignStr = new StringBuilder();
            for (int d = 0; d < D; d++) {
                assignStr.append(d).append("\n");
                assignStr.append(DirMult.output(docTopics[d])).append("\n");
                for (int n = 0; n < z[d].length; n++) {
                    assignStr.append(z[d][n]).append("\t");
                }
                assignStr.append("\n");
            }

            // output to a compressed file
            ArrayList<String> contentStrs = new ArrayList<>();
            contentStrs.add(modelStr.toString());
            contentStrs.add(assignStr.toString());
            contentStrs.add(billStr.toString());
            contentStrs.add(authorStr.toString());

            String filename = IOUtils.removeExtension(IOUtils.getFilename(filepath));
            ArrayList<String> entryFiles = new ArrayList<>();
            entryFiles.add(filename + ModelFileExt);
            entryFiles.add(filename + AssignmentFileExt);
            entryFiles.add(filename + BillFileExt);
            entryFiles.add(filename + AuthorFileExt);

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
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Excepion while inputing state from "
                    + filepath);
        }

        validate("Done reading state from " + filepath);
    }

    public void inputModel(String zipFilepath) {
        if (verbose) {
            logln("--- --- Loading model from " + zipFilepath);
        }

        try {
            String filename = IOUtils.removeExtension(IOUtils.getFilename(zipFilepath));
            BufferedReader reader = IOUtils.getBufferedReader(zipFilepath,
                    filename + ModelFileExt);
            wordWeights = new double[V];
            for (int vv = 0; vv < V; vv++) {
                String[] sline = reader.readLine().split("\t");
                int vIdx = Integer.parseInt(sline[0]);
                if (vv != vIdx) {
                    throw new MismatchRuntimeException(vIdx, vv);
                }
                wordWeights[vv] = Double.parseDouble(sline[1]);
            }

            topicWords = new DirMult[K];
            eta = new double[K];
            for (int kk = 0; kk < K; kk++) {
                int topicIdx = Integer.parseInt(reader.readLine());
                if (topicIdx != kk) {
                    throw new RuntimeException("Indices mismatch when loading model");
                }
                eta[kk] = Double.parseDouble(reader.readLine());
                topicWords[kk] = DirMult.input(reader.readLine());
            }

//            tau = new SparseVector[K];
//            for (int kk = 0; kk < K; kk++) {
//                tau[kk] = SparseVector.input(reader.readLine());
//            }

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
            for (int d = 0; d < D; d++) {
                int docIdx = Integer.parseInt(reader.readLine().split("\t")[0]);
                if (docIdx != d) {
                    throw new RuntimeException("Indices mismatch when loading assignments");
                }
                docTopics[d] = DirMult.input(reader.readLine());

                String[] sline = reader.readLine().split("\t");
                if (sline.length != words[d].length) {
                    throw new RuntimeException("[MISMATCH]. Doc "
                            + d + ". " + sline.length + " vs. " + words[d].length);
                }
                for (int n = 0; n < words[d].length; n++) {
                    z[d][n] = Integer.parseInt(sline[n]);
                }
            }
            reader.close();
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while inputing assignments from "
                    + zipFilepath);
        }
    }

    @Override
    public void outputTopicTopWords(File file, int numTopWords) {
        if (this.wordVocab == null) {
            throw new RuntimeException("The word vocab has not been assigned yet");
        }

        if (verbose) {
            logln("Outputing per-topic top words to " + file);
        }

        ArrayList<RankingItem<Integer>> sortedTopics = new ArrayList<RankingItem<Integer>>();
        for (int k = 0; k < K; k++) {
            sortedTopics.add(new RankingItem<Integer>(k, eta[k]));
        }
        Collections.sort(sortedTopics);

        try {
            BufferedWriter writer = IOUtils.getBufferedWriter(file);
            for (int ii = 0; ii < K; ii++) {
                int k = sortedTopics.get(ii).getObject();
                String topicLabel = "Topic " + k;
                if (labelVocab != null) {
                    topicLabel = labelVocab.get(k);
                }

                double[] distrs = topicWords[k].getDistribution();
                String[] topWords = getTopWords(distrs, numTopWords);
                writer.write("[" + topicLabel
                        + ", " + topicWords[k].getCountSum()
                        + ", " + MiscUtils.formatDouble(eta[k])
                        + "]");
                for (String topWord : topWords) {
                    writer.write("\t" + topWord);
                }

                writer.write("\n");

                int nonzero = 0;
                ArrayList<RankingItem<Integer>> rankLexs = new ArrayList<>();
//                for (int vv : tau[k].getIndices()) {
//                    rankLexs.add(new RankingItem<Integer>(vv, tau[k].get(vv)));
//                    nonzero++;
//                }
                Collections.sort(rankLexs);
                for (int jj = 0; jj < Math.min(10, rankLexs.size()); jj++) {
                    RankingItem<Integer> rankLex = rankLexs.get(jj);
                    if (rankLex.getPrimaryValue() > 0) {
                        writer.write("+++ " + wordVocab.get(rankLex.getObject())
                                + "\t" + MiscUtils.formatDouble(rankLex.getPrimaryValue())
                                + ", " + topicWords[k].getCount(rankLex.getObject())
                                + "\n");
                    }
                }

                for (int jj = 0; jj < Math.min(10, rankLexs.size()); jj++) {
                    RankingItem<Integer> rankLex = rankLexs.get(rankLexs.size() - 1 - jj);
                    if (rankLex.getPrimaryValue() < 0) {
                        writer.write("--- " + wordVocab.get(rankLex.getObject())
                                + "\t" + MiscUtils.formatDouble(rankLex.getPrimaryValue())
                                + ", " + topicWords[k].getCount(rankLex.getObject())
                                + "\n");
                    }
                }
                writer.write(">>> # word types with non-zero count: "
                        + topicWords[k].getSparseCounts().getIndices().size() + "\n");
                writer.write(">>> # non-zero tau's: " + nonzero + "\n\n");
            }
            writer.close();
        } catch (IOException e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while outputing topic words");
        }
    }

    class EtaDiffFunc implements DiffFunction {

        private final SparseVector[] za;

        public EtaDiffFunc() {
            za = new SparseVector[A];
            for (int aa = 0; aa < A; aa++) {
                za[aa] = new SparseVector(K);
            }
            for (int dd = 0; dd < D; dd++) {
                int aa = authors[dd];
                for (int kk : docTopics[dd].getSparseCounts().getIndices()) {
                    int count = docTopics[dd].getCount(kk);
                    za[aa].change(kk, (double) count / authorTotalWordWeights[aa]);
                }
            }
        }

        @Override
        public int domainDimension() {
            return K;
        }

        @Override
        public double valueAt(double[] w) {
            double llh = 0.0;
            for (int aa = 0; aa < A; aa++) {
                for (int kk = 0; kk < K; kk++) {
//                    double diff = us[aa][kk] - lexVals[aa].get(kk) - za[aa].get(kk) * w[kk];
                    double diff = us[aa][kk] - za[aa].get(kk) * w[kk];
                    llh += 0.5 * diff * diff / rho;
                }
            }
            double reg = 0.0;
            for (int kk = 0; kk < K; kk++) {
                reg += 0.5 * w[kk] * w[kk] / gamma;
            }
            return llh + reg;
        }

        @Override
        public double[] derivativeAt(double[] w) {
            double[] grads = new double[K];
            for (int aa = 0; aa < A; aa++) {
                double dotprod = za[aa].dotProduct(w);
                for (int kk : za[aa].getIndices()) {
//                    grads[kk] += (dotprod - us[aa][kk] + lexVals[aa].get(kk)) * za[aa].get(kk) / rho;
                    grads[kk] += (dotprod - us[aa][kk]) * za[aa].get(kk) / rho;
                }
            }
            for (int kk = 0; kk < K; kk++) {
                grads[kk] += w[kk] / gamma;
            }
            return grads;
        }
    }
//    class TauDiffFunc implements DiffFunction {
//
//        private final int kk;
//        private final SparseVector[] wa;
//
//        public TauDiffFunc(int topic) {
//            this.kk = topic;
//            this.wa = new SparseVector[A];
//            for (int aa = 0; aa < A; aa++) {
//                this.wa[aa] = new SparseVector(V);
//            }
//
//            for (int dd = 0; dd < D; dd++) {
//                int aa = authors[dd];
//                for (int nn = 0; nn < words[dd].length; nn++) {
//                    if (z[dd][nn] == kk) {
//                        this.wa[aa].change(words[dd][nn], 1.0 / authorTotalWordWeights[aa]);
//                    }
//                }
//            }
//        }
//
//        @Override
//        public int domainDimension() {
//            return V;
//        }
//
//        @Override
//        public double valueAt(double[] w) {
//            double llh = 0.0;
//            for (int aa = 0; aa < A; aa++) {
//                double diff = us[aa][kk] - topVals[aa].get(kk);
//                for (int vv : this.wa[aa].getIndices()) {
//                    diff -= this.wa[aa].get(vv) * w[kk];
//                }
//                llh += 0.5 * diff * diff / rho;
//            }
//            double reg = 0.0;
//            if (l2 > 0) {
//                for (int vv = 0; vv < V; vv++) {
//                    reg += 0.5 * l2 * w[vv] * w[vv];
//                }
//            }
//            return llh + reg;
//        }
//
//        @Override
//        public double[] derivativeAt(double[] w) {
//            double[] grads = new double[V];
//            for (int aa = 0; aa < A; aa++) {
//                double dotprod = wa[aa].dotProduct(w);
//                for (int vv : wa[aa].getIndices()) {
//                    grads[vv] += (dotprod - us[aa][kk] + topVals[aa].get(kk)) * wa[aa].get(kk) / rho;
//                }
//            }
//            if (l2 > 0) {
//                for (int vv = 0; vv < V; vv++) {
//                    grads[vv] += w[vv] * l2;
//                }
//            }
//
//            return grads;
//        }
//    }
}
