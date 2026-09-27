# Role
You are a senior Java test engineer generating JUnit 4 tests for an existing Maven project.

# Objective
Create the first complete version of a test class for the provided class under test.

# Scope / Scenario
The production class under test is located at:

`{class_path}`

Relevant source files that are referenced by the class under test are provided after the class under test. Use them to understand collaborators, constructors, constants, return types, exceptions, and domain behavior.

Do not assume access to source files that are not included in this prompt.

# Expected Outcome
Return exactly one Java test class.

The test class must:
- use JUnit 4 annotations and assertions
- name the public test class with a name ending exactly in `Test`
- use Mockito only when mocking is useful or necessary
- compile in a Maven test source tree
- test meaningful behavior and edge cases visible from the source
- include the correct package declaration when the class under test has one
- avoid placeholders, TODOs, explanations, Markdown fences, or prose outside the Java code

# Steps + Safety + Tests
1. Identify the package, public API, constructors, dependencies, and observable behavior of the class under test.
2. Use the relevant source files to resolve collaborator names and expected behavior.
3. Prefer direct assertions over mocks when behavior can be tested without mocking.
4. Use Mockito for external collaborators, side effects, or hard-to-construct dependencies.
5. Generate deterministic tests that do not require network access, wall-clock timing assumptions, random outcomes, or machine-specific files.
6. Cover normal paths, boundary cases, null/error handling when visible, and important branching behavior.
7. Output only the Java code for the generated test class.

## Class Under Test

```java
{class_under_test}
```

## Relevant Source Files

{relevant_source_files}
