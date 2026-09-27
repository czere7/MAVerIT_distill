package org.example;

import static org.junit.Assert.assertEquals;

import org.junit.Test;

/**
 * Fixture: what runs 1-4 actually produced. Valid, sensible JUnit 4 that simply never
 * ends -- the class is left open because generation hit the token cap mid-method rather
 * than emitting a stop token. Unbalanced braces, and it does not look terminated.
 */
public class Runaway {

    @Test
    public void firstCase() {
        assertEquals(1, 1);
    }

    @Test
    public void secondCase() {
        assertEquals(2, 2);
    }

    @Test
    public void thirdCaseCutShort() {
        assertEquals(3,
