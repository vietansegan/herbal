package labelestimator;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import sampler.labeled.baselines.TFIDFNN;
import util.RankingItem;
import util.SparseVector;
import util.StatUtils;

/**
 *
 * @author vietan
 */
public class TFIDFEstimator extends AbstractEstimator {

    private TFIDFNN tfidf;
    private final int N; // number of top word types
    private final double ratio;

    public TFIDFEstimator(int[][] words, int[] labels, int L, int V,
            int N, double ratio) {
        super(words, labels, L, V);
        this.N = N;
        this.ratio = ratio;
    }

    @Override
    public double[][] getPriors() {
        double[][] priors = new double[L + 1][V];

        // background topic
        int numTokens = 0;
        for (int[] word : words) {
            numTokens += word.length;
            for (int nn = 0; nn < word.length; nn++) {
                priors[L][word[nn]]++;
            }
        }
        for (int vv = 0; vv < V; vv++) {
            priors[L][vv] /= numTokens;
        }

        // estimate label priors
        this.tfidf = new TFIDFNN(words, labels, L, V, 5);
        this.tfidf.learn();
        SparseVector[] labelVectos = this.tfidf.getLabelVectors();
        for (int ll = 0; ll < L; ll++) {
            ArrayList<RankingItem<Integer>> rankItems = new ArrayList<>();
            for (int vv : labelVectos[ll].getIndices()) {
                rankItems.add(new RankingItem<Integer>(vv, labelVectos[ll].get(vv)));
            }
            Collections.sort(rankItems);

            Arrays.fill(priors[ll], 1.0);
            for (int ii = 0; ii < Math.min(N, rankItems.size()); ii++) {
                int vv = rankItems.get(ii).getObject();
                priors[ll][vv] += ratio;
            }
            double sum = StatUtils.sum(priors[ll]);
            for (int vv = 0; vv < V; vv++) {
                priors[ll][vv] /= sum;
            }
        }
        return priors;
    }
}
