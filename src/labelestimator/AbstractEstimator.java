package labelestimator;

/**
 *
 * @author vietan
 */
public abstract class AbstractEstimator {

    protected int[][] words;
    protected int[] labels;
    protected int L;
    protected int V;

    public AbstractEstimator(int[][] words, int[] labels, int L, int V) {
        this.words = words;
        this.labels = labels;
        this.L = L;
        this.V = V;
    }

    public abstract double[][] getPriors();
}
