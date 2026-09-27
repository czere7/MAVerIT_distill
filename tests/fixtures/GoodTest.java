package org.example;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

/**
 * Fixture: a well-formed JUnit 4 class with three test methods, exactly one of which
 * asserts nothing -- so count_test_without_assert has something to find.
 *
 * Note for maintainers: do not write the annotation name in this comment. The ported
 * counter splits on the bare string and would count the comment as a fourth method.
 */
public class GoodTest {

    @Test
    public void addsTwoNumbers() {
        assertEquals(4, 2 + 2);
    }

    @Test
    public void comparesStrings() {
        assertTrue("ab".startsWith("a"));
    }

    @Test
    public void executesWithoutAsserting() {
        Integer.parseInt("7");
    }
}
