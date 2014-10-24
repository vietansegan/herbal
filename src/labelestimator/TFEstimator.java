package labelestimator;

import util.StatUtils;

/**
 *
 * @author vietan
 */
public class TFEstimator extends AbstractEstimator {

    public TFEstimator(int[][] words, int[] labels, int L, int V) {
        super(words, labels, L, V);
    }

    @Override
    public double[][] getPriors() {
        // count
        double[][] priors = new double[L][V];
        double[] background = new double[V];
        for (int dd = 0; dd < words.length; dd++) {
            int label = labels[dd];
            for (int nn = 0; nn < words[dd].length; nn++) {
                background[words[dd][nn]]++;
                priors[label][words[dd][nn]]++;
            }
        }

        // normalize
        for (int ll = 0; ll < L; ll++) {
            double sum = StatUtils.sum(priors[ll]);
            for (int vv = 0; vv < V; vv++) {
                priors[ll][vv] /= sum;
            }
        }
        double sum = StatUtils.sum(background);
        for (int vv = 0; vv < V; vv++) {
            background[vv] /= sum;
        }

        // diff
        for (int vv = 0; vv < V; vv++) {
            for (int ll = 0; ll < L; ll++) {
                double diff = priors[ll][vv] - background[vv];
                priors[ll][vv] = diff < 0 ? 0 : diff;
            }
        }

        // renormalize
        for (int ll = 0; ll < L; ll++) {
            sum = StatUtils.sum(priors[ll]);
            for (int vv = 0; vv < V; vv++) {
                priors[ll][vv] /= sum;
            }
        }

        return priors;
    }
}
