package votepredictor.textidealpoint;

import java.io.BufferedReader;
import java.io.File;
import java.util.ArrayList;
import util.IOUtils;
import util.MismatchRuntimeException;
import util.SparseVector;
import votepredictor.BayesianIdealPoint;

/**
 *
 * @author vietan
 */
public abstract class AbstractTextSingleIdealPoint extends AbstractTextIdealPoint {

    public static final int numSteps = 10;
    // hyperparameters
    protected double rho;   // variance of authors' ideal points
    protected double sigma; // variance of votes' ideal points
    // ideal point
    protected double[] u; // [A]: authors' scores
    protected double[] x; // [B]
    protected double[] y; // [B]
    // internal
    protected int posAnchor;
    protected int negAnchor;

    public double[] getUs() {
        return this.u;
    }

    public double[] getXs() {
        return this.x;
    }

    public double[] getYs() {
        return this.y;
    }

    public abstract SparseVector[] predictOutMatrix();

    protected abstract void updateUs();

    protected double getLearningRate() {
        return 0.01;
    }

    /**
     * Update ideal point model's parameters using gradient ascent.
     *
     * @return Elapsed time
     */
    protected long updateUXY() {
        if (isReporting) {
            logln("+++ Updating UXY ...");
        }
        long sTime = System.currentTimeMillis();
        for (int step = 0; step < numSteps; step++) {
            updateUs();
            updateXYs();
        }
        long eTime = System.currentTimeMillis() - sTime;
        if (isReporting) {
            logln("--- --- time: " + eTime);
        }
        return eTime;
    }

    public void updateXYs() {
        double bRate = getLearningRate();
        for (int bb = 0; bb < B; bb++) {
            double gradX = 0.0;
            double gradY = 0.0;
            // likelihood
            for (int aa = 0; aa < A; aa++) {
                if (isValidVote(aa, bb)) {
                    double score = Math.exp(u[aa] * x[bb] + y[bb]);
                    gradX += u[aa] * (getVote(aa, bb) - score / (1 + score));
                    gradY += getVote(aa, bb) - score / (1 + score);
                }
            }
            // prior
            gradX -= x[bb] / sigma;
            gradY -= y[bb] / sigma;

            // update
            x[bb] += bRate * gradX;
            y[bb] += bRate * gradY;
        }
    }

    protected void initializeIdealPoint() {
        BayesianIdealPoint bip = new BayesianIdealPoint();
        bip.configure(1.0, 0.01, 50000, 0.0, sigma);
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
    }

    /**
     * Make prediction on held-out votes of known legislators and known bills,
     * averaging over multiple models.
     *
     * @return Predictions
     */
    public SparseVector[] predictInMatrixMultiples() {
        SparseVector[] predictions = null;
        int count = 0;
        File reportFolder = new File(this.getReportFolderPath());
        String[] files = reportFolder.list();
        for (String file : files) {
            if (!file.endsWith(".zip")) {
                continue;
            }
            this.inputState(new File(reportFolder, file));
            SparseVector[] partPreds = this.predictInMatrix();
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
        return predictions;
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
     * Input bill ideal points.
     *
     * @param zipFilepath File path
     */
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
            for (int bb = 0; bb < B; bb++) {
                x[bb] = Double.parseDouble(reader.readLine());
                y[bb] = Double.parseDouble(reader.readLine());
            }
            reader.close();
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while loading model from "
                    + zipFilepath);
        }
    }

    /**
     * Input author ideal points.
     *
     * @param zipFilepath File path
     */
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
     * Average vote predictions.
     *
     * @param predList List of predictions
     * @return Averaged predictions
     */
    protected SparseVector[] averagePredictions(ArrayList<SparseVector[]> predList) {
        SparseVector[] predictions = new SparseVector[validVotes.length];
        for (int aa = 0; aa < A; aa++) {
            int author = authorIndices.get(aa);
            predictions[author] = new SparseVector(validVotes[author].length);
            for (SparseVector[] pred : predList) {
                predictions[author].add(pred[author]);
            }
            predictions[author].scale(1.0 / predList.size());
        }
        return predictions;
    }
}
