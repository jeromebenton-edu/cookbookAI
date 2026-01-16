import { useEffect, useMemo, useRef, useState } from "react";
import { getAvailableOverlayPages, AvailablePagesResponse } from "../lib/api/parse";

type State = {
  loading: boolean;
  available: boolean;
  pages: number[];
  error?: string;
};

const cache: { data?: AvailablePagesResponse } = {};

export function useOverlayAvailability(pageNum: number) {
  const [state, setState] = useState<State>({ loading: true, available: false, pages: [] });
  const pageRef = useRef(pageNum);

  useEffect(() => {
    pageRef.current = pageNum;
  }, [pageNum]);

  useEffect(() => {
    let cancelled = false;
    async function load() {
      if (cache.data) {
        const available = cache.data.available_page_ids.includes(pageRef.current);
        setState({ loading: false, available, pages: cache.data.available_page_ids });
        return;
      }
      setState((prev) => ({ ...prev, loading: true, error: undefined }));
      try {
        const data = await getAvailableOverlayPages();
        cache.data = data;
        if (!cancelled) {
          setState({
            loading: false,
            available: data.available_page_ids.includes(pageRef.current),
            pages: data.available_page_ids
          });
        }
      } catch (err) {
        if (!cancelled) {
          // If availability cannot be determined, do not block the UI; assume available.
          setState({
            loading: false,
            available: true,
            pages: [],
            error: err instanceof Error ? err.message : "Failed to load availability"
          });
        }
      }
    }
    void load();
    return () => {
      cancelled = true;
    };
  }, [pageNum]);

  const helpers = useMemo(
    () => ({
      ...state,
      availablePageIds: state.pages,
      refresh: () => {
        cache.data = undefined;
        setState((prev) => ({ ...prev, loading: true }));
      }
    }),
    [state]
  );

  return helpers;
}
