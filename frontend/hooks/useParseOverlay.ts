import { useEffect, useMemo, useRef, useState } from "react";
import { getBostonPagePrediction, ParseResponse } from "../lib/api/parse";

type State = {
  data?: ParseResponse & { image_url_resolved: string };
  loading: boolean;
  error?: string;
  unavailable?: boolean;
  hint?: string;
};

export function useParseOverlay(pageNum: number, enabled: boolean) {
  const cacheRef = useRef<Record<number, ParseResponse & { image_url_resolved: string }>>({});
  const [state, setState] = useState<State>({ loading: false });

  const fetchData = async (forceRefresh = false) => {
    if (!enabled || !pageNum) return;
    if (!forceRefresh && cacheRef.current[pageNum]) {
      setState({ loading: false, data: cacheRef.current[pageNum] });
      return;
    }
    setState((prev) => ({ ...prev, loading: true, error: undefined }));
    try {
      const res = await getBostonPagePrediction(pageNum, { refresh: forceRefresh });
      cacheRef.current[pageNum] = res;
      setState({ loading: false, data: res, unavailable: false, hint: undefined });
    } catch (err) {
      const status = (err as any)?.status;
      const body = (err as any)?.body;
      const isUnavailable =
        status === 404 && body && typeof body === "object" && body.error === "overlay_not_available";
      const message =
        (body && typeof body?.message === "string" && body.message) ||
        (err instanceof Error ? err.message : "Unknown error");
      setState({
        loading: false,
        error: message,
        unavailable: isUnavailable,
        hint: typeof body?.hint === "string" ? body.hint : undefined
      });
    }
  };

  useEffect(() => {
    if (enabled) {
      void fetchData(false);
    }
  }, [enabled, pageNum]);

  return useMemo(
    () => ({
      data: state.data,
      loading: state.loading,
      error: state.error,
      unavailable: state.unavailable,
      hint: state.hint,
      refresh: () => fetchData(true)
    }),
    [state]
  );
}
