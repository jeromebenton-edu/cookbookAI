import assert from "node:assert/strict";
import { test } from "node:test";
import { getBundledExamples, getDefaultExample, getExampleById } from "../lib/bundledExamples";

test("getBundledExamples - should return an array of examples", () => {
  const examples = getBundledExamples();
  assert.ok(Array.isArray(examples));
  assert.ok(examples.length > 0);
});

test("getBundledExamples - should have valid structure for each example", () => {
  const examples = getBundledExamples();
  examples.forEach((example) => {
    // Check properties exist
    assert.ok("id" in example);
    assert.ok("pageImage" in example);
    assert.ok("prediction" in example);
    assert.ok("meta" in example);

    // Check types
    assert.equal(typeof example.id, "string");
    assert.equal(typeof example.pageImage, "string");
    assert.equal(typeof example.prediction, "object");
    assert.equal(typeof example.meta, "object");

    // Check prediction structure
    assert.ok("title" in example.prediction);
    assert.ok("ingredients" in example.prediction);
    assert.ok("instructions" in example.prediction);
    assert.ok("confidence" in example.prediction);

    // Check meta structure
    assert.ok("title" in example.meta);
    assert.ok("difficulty" in example.meta);
  });
});

test("getBundledExamples - should have unique IDs", () => {
  const examples = getBundledExamples();
  const ids = examples.map((ex) => ex.id);
  const uniqueIds = new Set(ids);
  assert.equal(uniqueIds.size, ids.length);
});

test("getDefaultExample - should return the first example", () => {
  const defaultExample = getDefaultExample();
  const allExamples = getBundledExamples();

  assert.deepEqual(defaultExample, allExamples[0]);
});

test("getDefaultExample - should have a valid prediction", () => {
  const defaultExample = getDefaultExample();

  assert.ok(defaultExample.prediction.title);
  assert.ok(defaultExample.prediction.ingredients.length > 0);
  assert.ok(defaultExample.prediction.instructions.length > 0);
});

test("getDefaultExample - should have high confidence values", () => {
  const defaultExample = getDefaultExample();
  const conf = defaultExample.prediction.confidence;

  assert.ok(conf.overall > 0.7);
  assert.ok(conf.overall <= 1.0);
});

test("getExampleById - should return example for valid ID", () => {
  const examples = getBundledExamples();
  const firstId = examples[0].id;

  const found = getExampleById(firstId);
  assert.ok(found !== null);
  assert.equal(found?.id, firstId);
});

test("getExampleById - should return null for invalid ID", () => {
  const found = getExampleById("nonexistent_id");
  assert.equal(found, null);
});
