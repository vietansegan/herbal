package votepredictor.textidealpoint;

import java.io.BufferedReader;
import util.IOUtils;
import util.MiscUtils;
import util.MismatchRuntimeException;
import util.SparseVector;

/**
 *
 * @author vietan
 */
public abstract class AbstractTextMultipleIdealPoint extends AbstractTextIdealPoint {

    protected double rho;   // variance of authors' ideal points
    protected double sigma; // variance of votes' ideal points
    protected double[][] xs; // [B][K + 1]
    protected double[][] us; // [A][K]

    protected int K;

    public double[][] getUs() {
        return this.us;
    }

    public double[][] getXs() {
        return this.xs;
    }

    public SparseVector[] predictInMatrix() {
        SparseVector[] predictions = new SparseVector[validVotes.length];
        for (int aa = 0; aa < A; aa++) {
            int author = authorIndices.get(aa);
            predictions[author] = new SparseVector(validVotes[author].length);
            for (int bb = 0; bb < B; bb++) {
                int bill = billIndices.get(bb);
                if (isValidVote(aa, bb)) {
                    double dotprod = xs[bb][K];
                    for (int kk = 0; kk < K; kk++) {
                        dotprod += us[aa][kk] * xs[bb][kk];
                    }
                    double score = Math.exp(dotprod);
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
            xs = new double[B][];
            for (int bb = 0; bb < B; bb++) {
                xs[bb] = MiscUtils.stringToDoubleArray(reader.readLine());
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
}
