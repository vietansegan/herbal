package votepredictor;

import votepredictor.textidealpoint.AbstractTextIdealPoint;
import cc.mallet.optimize.LimitedMemoryBFGS;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import optimization.OWLQNLinearRegression;
import optimization.RidgeLinearRegressionLBFGS;
import sampling.util.SparseCount;
import util.IOUtils;
import util.MiscUtils;
import util.MismatchRuntimeException;
import util.RankingItem;
import util.SamplerUtils;
import util.SparseVector;
import util.evaluation.Measurement;

/**
 *
 * @author vietan
 */
public class LexicalIdealPoint extends AbstractTextIdealPoint {

    public double sigma;
    public double rho;
    public double lambda;

    public double l1;
    public double l2;

    protected double[] u; // [A]: authors' scores
    protected double[] x; // [B]
    protected double[] y; // [B]
    protected double[] tau; // [V]
    protected double[] idfs;
    protected boolean tfidf; // configure

    // derive
    protected SparseVector[] authorVectors;

    public LexicalIdealPoint() {
        this.basename = "Lexical-ideal-point";
    }

    public LexicalIdealPoint(String bname) {
        this.basename = bname;
    }

    public boolean hasTfIdf() {
        return this.tfidf;
    }

    public double[] getIdfs() {
        return this.idfs;
    }

    @Override
    protected void prepareDataStatistics() {
        super.prepareDataStatistics();

        if (this.tfidf) {
            if (this.idfs == null) { // training
                this.idfs = MiscUtils.getIDFs(words, V);
            } // else, idfs are loaded from external file
        }

        if (!this.tfidf && this.idfs != null) {
            throw new RuntimeException("Not using TF-IDF");
        }

        this.authorVectors = new SparseVector[A];
        for (int aa = 0; aa < A; aa++) {
            this.authorVectors[aa] = new SparseVector(V);
        }
        for (int dd = 0; dd < D; dd++) {
            int aa = authors[dd];
            SparseCount counts = new SparseCount();
            for (int nn = 0; nn < words[dd].length; nn++) {
                counts.increment(words[dd][nn]);
            }

            for (int vv : counts.getIndices()) {
                int count = counts.getCount(vv);
                if (this.tfidf) {
                    this.authorVectors[aa].change(vv, Math.log(count + 1) * idfs[vv]);
                } else {
                    this.authorVectors[aa].change(vv, count);
                }
            }
        }

        for (int aa = 0; aa < A; aa++) {
            this.authorVectors[aa].normalize();
        }
    }

    public void configure(
            String folder, int V, double rho, double sigma, double lambda,
            int maxiter, boolean tfidf) {
        if (verbose) {
            logln("Configuring ...");
        }

        this.folder = folder;
        this.V = V;

        this.hyperparams = new ArrayList<Double>();
        this.rho = rho;
        this.sigma = sigma;
        this.lambda = lambda;
        this.MAX_ITER = maxiter;
        this.tfidf = tfidf;
        this.wordWeightType = WordWeightType.NONE;
        this.sampledParams = new ArrayList<ArrayList<Double>>();
        this.sampledParams.add(cloneHyperparameters());
        this.report = true;

        StringBuilder str = new StringBuilder();
        str.append(basename)
                .append("_M-").append(MAX_ITER)
                .append("_r-").append(formatter.format(rho))
                .append("_s-").append(formatter.format(sigma))
                .append("_l-").append(formatter.format(lambda))
                .append("_tfidf-").append(tfidf);
        this.name = str.toString();

        if (verbose) {
            logln("--- folder\t" + folder);
            logln("--- num word types:\t" + V);
            logln("--- rho:\t" + MiscUtils.formatDouble(rho));
            logln("--- reg sigma:\t" + MiscUtils.formatDouble(sigma));
            logln("--- lambda:\t" + MiscUtils.formatDouble(lambda));
            logln("--- max-iters:\t" + MAX_ITER);
        }
    }

    public void configure(
            String folder, int V, double rho, double sigma, double l1, double l2,
            int maxiter, boolean tfidf) {
        if (verbose) {
            logln("Configuring ...");
        }

        this.folder = folder;
        this.V = V;

        this.hyperparams = new ArrayList<Double>();
        this.rho = rho;
        this.sigma = sigma;
        this.l1 = l1;
        this.l2 = l2;
        this.MAX_ITER = maxiter;
        this.tfidf = tfidf;
        this.wordWeightType = WordWeightType.NONE;
        this.sampledParams = new ArrayList<ArrayList<Double>>();
        this.sampledParams.add(cloneHyperparameters());
        this.report = true;

        StringBuilder str = new StringBuilder();
        str.append(basename)
                .append("_M-").append(MAX_ITER)
                .append("_r-").append(formatter.format(rho))
                .append("_s-").append(formatter.format(sigma))
                .append("_l1-").append(MiscUtils.formatDouble(l1, 10))
                .append("_l2-").append(MiscUtils.formatDouble(l2, 10))
                .append("_tfidf-").append(tfidf);
        this.name = str.toString();

        if (verbose) {
            logln("--- folder\t" + folder);
            logln("--- num word types:\t" + V);
            logln("--- rho:\t" + MiscUtils.formatDouble(rho));
            logln("--- reg sigma:\t" + MiscUtils.formatDouble(sigma));
            logln("--- l1:\t" + MiscUtils.formatDouble(l1, 5));
            logln("--- l2:\t" + MiscUtils.formatDouble(l2, 5));
            logln("--- max-iters:\t" + MAX_ITER);
        }
    }

    public double[] getUs() {
        return this.u;
    }

    public double[] getXs() {
        return this.x;
    }

    public double[] getYs() {
        return this.y;
    }

    public double[] getTaus() {
        return this.tau;
    }

    public SparseVector[] getAuthorLexicalVectors() {
        return this.authorVectors;
    }

    public void setupData(int[][] votes, ArrayList<Integer> authorIndices,
            ArrayList<Integer> billIndices, boolean[][] validVotes,
            SparseVector[] authorVectors) {
        this.votes = votes;
        this.authorIndices = authorIndices;
        this.billIndices = billIndices;
        this.validVotes = validVotes;
        this.authorVectors = authorVectors;
    }

    public void setAuthorLexicalVectors(SparseVector[] authorVectors) {
        this.authorVectors = authorVectors;
    }

    public double[] getPredictedUs() {
        double[] predUs = new double[A];
        for (int aa = 0; aa < A; aa++) {
            predUs[aa] = authorVectors[aa].dotProduct(tau);
        }
        return predUs;
    }

    /**
     * Make prediction on held-out votes of known legislators and known votes.
     *
     * @return Predicted probabilities
     */
    public SparseVector[] predictInMatrix() {
        SparseVector[] predictions = new SparseVector[validVotes.length];
        for (int aa = 0; aa < A; aa++) {
            int author = authorIndices.get(aa);
            predictions[author] = new SparseVector(validVotes[author].length);
            for (int bb = 0; bb < B; bb++) {
                int bill = billIndices.get(bb);
                if (isValidVote(aa, bb)) {
                    double score = Math.exp(u[aa] * x[bb] + y[bb]);
                    double prob = score / (1.0 + score);
                    predictions[author].set(bill, prob);
                }
            }
        }
        return predictions;
    }

    /**
     * Make predictions on held-out votes of unknown legislators on known votes.
     *
     * @return Predicted probabilities
     */
    public SparseVector[] predictOutMatrix() {
        SparseVector[] predictions = new SparseVector[validVotes.length];
        for (int aa = 0; aa < A; aa++) {
            int author = authorIndices.get(aa);
            predictions[author] = new SparseVector(validVotes[author].length);
            for (int bb = 0; bb < B; bb++) {
                int bill = billIndices.get(bb);
                if (isValidVote(aa, bb)) {
                    double score = Math.exp(authorVectors[aa].dotProduct(tau) * x[bb] + y[bb]);
                    double prob = score / (1.0 + score);
                    predictions[author].set(bill, prob);
                }
            }
        }
        return predictions;
    }

    @Override
    public void initialize() {
        BayesianIdealPoint bip = new BayesianIdealPoint();
        bip.configure(1.0, 0.01, MAX_ITER, 0.0, sigma);
        bip.setTrain(votes, authorIndices, billIndices, validVotes);

        File bipFolder = new File(folder, bip.getName());
        File bipFile = new File(bipFolder, ModelFile);

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
        this.y = bip.getYs();

        this.tau = new double[V];
        for (int vv = 0; vv < V; vv++) {
            this.tau[vv] = SamplerUtils.getGaussian(0.0, 5.0);
        }
    }

    @Override
    public void iterate() {
        if (log && !isLogging()) {
            IOUtils.createFolder(getSamplerFolderPath());
            openLogger();
        }

        logln(getClass().toString());
        startTime = System.currentTimeMillis();

        if (lambda > 0) {
            updateTauLBFGS();
        } else {
            updateTauOWLQN();
        }

        if (verbose) {
            logln("--- Evaluating ...");
            SparseVector[] predictions = predictInMatrix();
            ArrayList<Measurement> measurements = AbstractVotePredictor
                    .evaluate(votes, validVotes, predictions);
            for (Measurement m : measurements) {
                logln(">>> i >>> " + m.getName() + ": " + m.getValue());
            }

            predictions = predictOutMatrix();
            measurements = AbstractVotePredictor
                    .evaluate(votes, validVotes, predictions);
            for (Measurement m : measurements) {
                logln(">>> o >>> " + m.getName() + ": " + m.getValue());
            }
        }

        float ellapsedSeconds = (System.currentTimeMillis() - startTime) / (1000);
        logln("Total runtime iterating: " + ellapsedSeconds + " seconds");

        if (log && isLogging()) {
            closeLogger();
        }
    }

    public double getLearningRate() {
        return 0.01;
    }

    public long updateTauLBFGS() {
        if (verbose) {
            logln("+++ Updating lexical regression parameters ...");
        }
        long sTime = System.currentTimeMillis();

        RidgeLinearRegressionLBFGS optimizable = new RidgeLinearRegressionLBFGS(
                u, tau, authorVectors, rho, 0.0, lambda);

        LimitedMemoryBFGS optimizer = new LimitedMemoryBFGS(optimizable);
        boolean converged = false;
        try {
            converged = optimizer.optimize();
        } catch (Exception ex) {
            ex.printStackTrace();
        }

        // update regression parameters
        for (int vv = 0; vv < V; vv++) {
            tau[vv] = optimizable.getParameter(vv);
        }

        if (verbose) {
            logln("--- converged? " + converged);
            logln("--- MSE: " + getMSE());
        }

        long eTime = System.currentTimeMillis() - sTime;
        if (verbose) {
            logln("--- --- time: " + eTime);
        }
        return eTime;
    }

    public long updateTauOWLQN() {
        if (verbose) {
            logln("+++ Updating lexical regression parameters using OWL-QN ...");
        }
        long sTime = System.currentTimeMillis();

        OWLQNLinearRegression opt = new OWLQNLinearRegression(basename, l1, l2);
        opt.train(authorVectors, u, V);
        this.tau = opt.getWeights();

        long eTime = System.currentTimeMillis() - sTime;
        if (verbose) {
            logln("--- --- time: " + eTime);
        }
        return eTime;
    }

    private double getMSE() {
        double mse = 0.0;
        for (int aa = 0; aa < A; aa++) {
            double diff = authorVectors[aa].dotProduct(tau) - u[aa];
            mse += diff * diff;
        }
        return mse / A;
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

    }

    public void outputAuthorFeatures(File file) {
        if (verbose) {
            logln("--- Outputing author features to " + file);
        }
        try {
            SparseVector[] authorFeatures = new SparseVector[A];
            for (int aa = 0; aa < A; aa++) {
                authorFeatures[aa] = new SparseVector(1);
                authorFeatures[aa].set(0, authorVectors[aa].dotProduct(tau));
            }
            BufferedWriter writer = IOUtils.getBufferedWriter(file);
            writer.write(A + "\n");
            for (int aa = 0; aa < A; aa++) {
                writer.write(SparseVector.output(authorFeatures[aa]) + "\n");
            }
            writer.close();
        } catch (IOException e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while outputing to " + file);
        }
    }

    public SparseVector[] inputAuthorFeatures(File file) {
        SparseVector[] authorFeatures = null;
        if (verbose) {
            logln("--- Inputing author features from " + file);
        }
        try {
            BufferedReader reader = IOUtils.getBufferedReader(file);
            int numA = Integer.parseInt(reader.readLine());
            if (numA != A) {
                throw new MismatchRuntimeException(numA, A);
            }
            authorFeatures = new SparseVector[A];
            for (int aa = 0; aa < A; aa++) {
                authorFeatures[aa] = SparseVector.input(reader.readLine());
            }
            reader.close();
        } catch (IOException | NumberFormatException | MismatchRuntimeException e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while inputing from " + file);
        }
        return authorFeatures;
    }

    @Override
    public void outputState(String filepath) {
        if (verbose) {
            logln("--- Outputing current state to " + filepath);
        }
        try {
            StringBuilder dataStr = new StringBuilder();
            dataStr.append(A).append("\n");
            for (int aa = 0; aa < A; aa++) {
                dataStr.append(u[aa]).append("\n");
            }

            StringBuilder modelStr = new StringBuilder();
            modelStr.append(B).append("\n");
            for (int bb = 0; bb < B; bb++) {
                modelStr.append(x[bb]).append("\t").append(y[bb]).append("\n");
            }
            modelStr.append(V).append("\n");
            for (int vv = 0; vv < V; vv++) {
                modelStr.append(tau[vv]).append("\n");
            }
            if (this.tfidf) {
                for (int vv = 0; vv < V; vv++) {
                    modelStr.append(idfs[vv]).append("\n");
                }
            }

            ArrayList<String> contentStrs = new ArrayList<>();
            contentStrs.add(modelStr.toString());
            contentStrs.add(dataStr.toString());

            String filename = IOUtils.removeExtension(IOUtils.getFilename(filepath));
            ArrayList<String> entryFiles = new ArrayList<>();
            entryFiles.add(filename + ModelFileExt);
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
            inputAuthorScore(filepath);
            inputModel(filepath);
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

            // bills
            int numB = Integer.parseInt(reader.readLine());
            if (numB != B) {
                throw new MismatchRuntimeException(numB, B);
            }
            this.x = new double[B];
            this.y = new double[B];
            for (int bb = 0; bb < B; bb++) {
                String[] sline = reader.readLine().split("\t");
                x[bb] = Double.parseDouble(sline[0]);
                y[bb] = Double.parseDouble(sline[1]);
            }
            // tau
            int numV = Integer.parseInt(reader.readLine());
            if (numV != V) {
                throw new MismatchRuntimeException(numV, V);
            }
            this.tau = new double[V];
            for (int vv = 0; vv < V; vv++) {
                this.tau[vv] = Double.parseDouble(reader.readLine());
            }

            if (this.tfidf) {
                this.idfs = new double[V];
                for (int vv = 0; vv < V; vv++) {
                    this.idfs[vv] = Double.parseDouble(reader.readLine());
                }
            }

            reader.close();
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while inputing model from "
                    + zipFilepath);
        }
    }

    public void inputAuthorScore(String zipFilepath) {
        if (verbose) {
            logln("--- --- Loading author scores from " + zipFilepath);
        }
        try {
            String filename = IOUtils.removeExtension(IOUtils.getFilename(zipFilepath));
            BufferedReader reader = IOUtils.getBufferedReader(zipFilepath, filename + AuthorFileExt);
            // authors
            int numA = Integer.parseInt(reader.readLine());
            if (numA != A) {
                throw new MismatchRuntimeException(numA, A);
            }
            this.u = new double[A];
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

    public void outputLexicalRegressionParameters(File file, ArrayList<String> vocab) {
        if (verbose) {
            logln("Outputing to " + file);
        }
        try {
            ArrayList<RankingItem<Integer>> rankLex = new ArrayList<>();
            for (int vv = 0; vv < V; vv++) {
                rankLex.add(new RankingItem<Integer>(vv, tau[vv]));
            }
            Collections.sort(rankLex);

            BufferedWriter writer = IOUtils.getBufferedWriter(file);
            for (int vv = 0; vv < V; vv++) {
                RankingItem<Integer> item = rankLex.get(vv);
                writer.write(item.getObject()
                        + "\t" + vocab.get(item.getObject())
                        + "\t" + item.getPrimaryValue()
                        + "\n");
            }
            writer.close();
        } catch (IOException e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while outputing to " + file);
        }
    }
}
