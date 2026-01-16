import assert from "node:assert/strict";
import { test } from "node:test";
import { evaluateDemoPageSelection } from "../lib/demoSelection";

test("uses backend default when no query is provided", () => {
  const result = evaluateDemoPageSelection({
    queryPageId: null,
    defaultPageId: 69,
    featuredPages: [{ page_id: 69, is_recipe_page: true }],
  });
  assert.equal(result.initialPageId, 69);
  assert.equal(result.canonicalPageId, 69);
  assert.equal(result.redirectPageId, null);
  assert.equal(result.isFrontMatter, false);
  assert.equal(result.isOutOfFeatured, false);
});

test("honors query param when it points to a recipe page", () => {
  const result = evaluateDemoPageSelection({
    queryPageId: 12,
    defaultPageId: 69,
    featuredPages: [
      { page_id: 12, is_recipe_page: true },
      { page_id: 69, is_recipe_page: true },
    ],
  });
  assert.equal(result.initialPageId, 12);
  assert.equal(result.canonicalPageId, null);
  assert.equal(result.redirectPageId, null);
  assert.equal(result.isFrontMatter, false);
  assert.equal(result.isOutOfFeatured, false);
});

test("redirects front matter queries back to the default recipe page", () => {
  const result = evaluateDemoPageSelection({
    queryPageId: 4,
    defaultPageId: 69,
    featuredPages: [
      { page_id: 4, is_recipe_page: false },
      { page_id: 69, is_recipe_page: true },
    ],
  });
  assert.equal(result.initialPageId, 69);
  assert.equal(result.redirectPageId, 69);
  assert.equal(result.isFrontMatter, true);
  assert.equal(result.isOutOfFeatured, false);
  assert.equal(result.attemptedPageId, 4);
});

test("redirects queries that are not featured pages back to default", () => {
  const result = evaluateDemoPageSelection({
    queryPageId: 41,
    defaultPageId: 69,
    featuredPages: [{ page_id: 69, is_recipe_page: true }],
  });
  assert.equal(result.initialPageId, 69);
  assert.equal(result.redirectPageId, 69);
  assert.equal(result.isFrontMatter, false);
  assert.equal(result.isOutOfFeatured, true);
  assert.equal(result.attemptedPageId, 41);
});
