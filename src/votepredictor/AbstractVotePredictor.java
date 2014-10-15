package votepredictor;

import core.AbstractModel;
import data.Vote;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.Set;
import util.SparseVector;
import util.evaluation.Measurement;
import util.evaluation.RankingEvaluation;

/**
 *
 * @author vietan
 */
public abstract class AbstractVotePredictor extends AbstractModel {

    public AbstractVotePredictor(String name) {
        super(name);
    }

    /**
     * Evaluate predicted values for each vote.
     *
     * @param votes Binary matrix storing votes (some votes can be missing)
     * @param testVotes Boolean matrix indicating valid votes
     * @param predictedValues Predicted values
     * @return List of evaluation measurements
     */
    public static ArrayList<Measurement> evaluate(
            int[][] votes,
            boolean[][] testVotes,
            SparseVector[] predictedValues) {
        ArrayList<Measurement> measurements = new ArrayList<>();

        double llh = 0.0;
        int count = 0;
        Set<String> withVotes = new HashSet<>();
        ArrayList<String> voteList = new ArrayList<>();
        ArrayList<Double> voteScores = new ArrayList<>();
        for (int aa = 0; aa < votes.length; aa++) {
            for (int vv = 0; vv < votes[aa].length; vv++) {
                if (!testVotes[aa][vv]) {
                    continue;
                }
                String key = aa + "_" + vv;
                double val = predictedValues[aa].get(vv);
                voteList.add(key);
                voteScores.add(val);

                if (votes[aa][vv] == Vote.WITH) {
                    withVotes.add(key);
                    llh += Math.log(val);
                } else if (votes[aa][vv] == Vote.AGAINST) {
                    llh += Math.log(1.0 - val);
                } else {
                    throw new RuntimeException("Missing data");
                }
                count++;
            }
        }

        // compute log likelihood of test data
        measurements.add(new Measurement("count", count));
        measurements.add(new Measurement("loglikelihood", llh));
        measurements.add(new Measurement("avg-loglikelihood", llh / count));

        // compute ranking performances
        double[] scores = new double[voteScores.size()];
        Set<Integer> withIndices = new HashSet<>();
        for (int ii = 0; ii < scores.length; ii++) {
            scores[ii] = voteScores.get(ii);

            String vote = voteList.get(ii);
            if (withVotes.contains(vote)) {
                withIndices.add(ii);
            }
        }
        RankingEvaluation withRankPerf = new RankingEvaluation(scores, withIndices);
        withRankPerf.computeAUCs();
        withRankPerf.computePRF();
        for(Measurement m : withRankPerf.getMeasurements()) {
            measurements.add(m);
        }
        return measurements;
    }
}
