package votepredictor;

import core.AbstractModel;
import data.Vote;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.Set;
import util.IOUtils;
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
     * Output the predicted scores for each test vote.
     *
     * @param predFile Output file
     * @param votes True votes (optional)
     * @param predictions Each sparse vector corresponds to a voter. The key of
     * the sparse vector is the vote index and the value is the predicted
     * probability that the corresponding vote is 1.
     */
    public static void outputPredictions(File predFile, int[][] votes,
            SparseVector[] predictions) {
        System.out.println("Outputing predictions to " + predFile);
        try {
            BufferedWriter writer = IOUtils.getBufferedWriter(predFile);
            writer.write(predictions.length + "\n");
            for (int aa = 0; aa < predictions.length; aa++) {
                if (predictions[aa] == null || predictions[aa].isEmpty()) {
                    continue;
                }
                writer.write(Integer.toString(aa));
                for (int key : predictions[aa].getIndices()) {
                    writer.write("\t" + key
                            + ":" + predictions[aa].get(key));
                    if (votes != null) {
                        writer.write(":" + votes[aa][key]);
                    }
                }
                writer.write("\n");
            }
            writer.close();
        } catch (IOException e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while outputing predictions to "
                    + predFile);
        }
    }

    public static SparseVector[] inputPredictions(File predFile) {
        SparseVector[] predictions = null;
        try {
            BufferedReader reader = IOUtils.getBufferedReader(predFile);
            int numPreds = Integer.parseInt(reader.readLine());
            predictions = new SparseVector[numPreds];
            for (int ii = 0; ii < numPreds; ii++) {
                predictions[ii] = new SparseVector();
            }
            String line;
            while ((line = reader.readLine()) != null) {
                String[] sline = line.split("\t");
                int aa = Integer.parseInt(sline[0]);
                for (int jj = 1; jj < sline.length; jj++) {
                    int key = Integer.parseInt(sline[jj].split(":")[0]);
                    double pred = Double.parseDouble(sline[jj].split(":")[1]);
                    predictions[aa].set(key, pred);
                }
            }
            reader.close();
        } catch (IOException e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while inputing predictions from "
                    + predFile);
        }
        return predictions;
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
        for (Measurement m : withRankPerf.getMeasurements()) {
            measurements.add(m);
        }
        return measurements;
    }
}
