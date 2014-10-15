package votepredictor;

import java.io.File;
import java.util.Random;
import main.GlobalConstants;
import util.SparseVector;

/**
 *
 * @author vietan
 */
public class RandomPredictor extends AbstractVotePredictor {

    private final Random rand;

    public RandomPredictor(String name) {
        super(name);
        this.rand = new Random(GlobalConstants.RANDOM_SEED);
    }

    /**
     * Predict the probability of a vote using a random number.
     *
     * @param testVotes Boolean matrix specifying the test votes
     * @return An array of sparse vectors, each belongs to an author
     */
    public SparseVector[] test(boolean[][] testVotes) {
        SparseVector[] predictions = new SparseVector[testVotes.length];
        for (int aa = 0; aa < predictions.length; aa++) {
            predictions[aa] = new SparseVector();
            for (int vv = 0; vv < testVotes[aa].length; vv++) {
                if (testVotes[aa][vv]) {
                    predictions[aa].set(vv, rand.nextDouble());
                }
            }
        }
        return predictions;
    }

    @Override
    public void output(File modelFile) {

    }

    @Override
    public void input(File modelFile) {

    }
}
