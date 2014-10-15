package data;

import core.AbstractObject;

/**
 *
 * @author vietan
 */
public class Vote extends AbstractObject<String> {

    public static final int AGAINST = 0;
    public static final int WITH = 1;
    public static final int MISSING = -1;

    public Vote(String id) {
        super(id);
    }
}
