/**
 * Controlled Drift Simulation Scenarios for the Drift Monitor Workspace.
 * Accurately mirrors Phase 6 research scenarios for live SOC demonstration.
 */

import { SAMPLE_BENIGN_FLOW } from "./sample-data.ts";
import type { NetworkFlowRecord } from "../types/api.ts";

export interface DriftScenario {
  id: string;
  name: string;
  category: "control" | "covariate" | "categorical" | "pipeline" | "adversarial";
  description: string;
  targetFeatures: string[];
  expectedDriftLevel: "STABLE" | "LOW_DRIFT" | "MODERATE_DRIFT" | "CRITICAL_DRIFT";
  expectedAction: string;
  operationalRationale: string;
  generateBatch: () => NetworkFlowRecord[];
}

/**
 * Creates 35 realistic flows with pseudo-random jitter around the base template.
 */
function createJitteredBatch(
  base: NetworkFlowRecord,
  count: number = 35,
  modifier?: (flow: NetworkFlowRecord, index: number) => NetworkFlowRecord
): NetworkFlowRecord[] {
  const result: NetworkFlowRecord[] = [];
  for (let i = 0; i < count; i++) {
    // Deterministic pseudo-jitter using sine wave
    const jitter = 1.0 + 0.08 * Math.sin(i * 1.7);
    const flow: NetworkFlowRecord = {
      ...base,
      dur: Math.max(0.001, parseFloat((base.dur * jitter).toFixed(4))),
      spkts: Math.max(1, Math.round(base.spkts * jitter)),
      dpkts: Math.max(1, Math.round(base.dpkts * jitter)),
      sbytes: Math.max(50, Math.round(base.sbytes * jitter)),
      dbytes: Math.max(50, Math.round(base.dbytes * jitter)),
      rate: Math.max(1.0, parseFloat((base.rate * jitter).toFixed(2))),
      sload: Math.max(100.0, parseFloat((base.sload * jitter).toFixed(2))),
      dload: Math.max(100.0, parseFloat((base.dload * jitter).toFixed(2))),
      sinpkt: Math.max(0.1, parseFloat((base.sinpkt * jitter).toFixed(2))),
      dinpkt: Math.max(0.1, parseFloat((base.dinpkt * jitter).toFixed(2))),
    };

    if (modifier) {
      result.push(modifier(flow, i));
    } else {
      result.push(flow);
    }
  }
  return result;
}

export const DRIFT_SCENARIOS: DriftScenario[] = [
  {
    id: "scenario_0_stable_control",
    name: "Scenario 0: Stable Reference Control Batch",
    category: "control",
    description: "In-distribution traffic closely matching baseline training statistics. All metrics remain within normal variance bounds.",
    targetFeatures: ["None (Control)"],
    expectedDriftLevel: "STABLE",
    expectedAction: "CONTINUE_MONITORING",
    operationalRationale: "No feature divergence detected (all PSI < 0.10). Telemetry reflects standard production operating conditions.",
    generateBatch: () => createJitteredBatch(SAMPLE_BENIGN_FLOW, 35),
  },
  {
    id: "scenario_1_numerical_shift",
    name: "Scenario 1: Transport Volume & Duration Shift",
    category: "covariate",
    description: "Simulated 4x-5x scaling on transport volume variables (sbytes, dur, sload, rate).",
    targetFeatures: ["sbytes", "dur", "sload", "rate", "spkts"],
    expectedDriftLevel: "CRITICAL_DRIFT",
    expectedAction: "RECOMMEND_OFFLINE_EVALUATION",
    operationalRationale: "Systemic multivariate drift across core bandwidth metrics. Flags features and suggests offline validation.",
    generateBatch: () =>
      createJitteredBatch(SAMPLE_BENIGN_FLOW, 35, (flow) => ({
        ...flow,
        dur: parseFloat((flow.dur * 4.2).toFixed(4)),
        sbytes: Math.round(flow.sbytes * 4.8),
        sload: parseFloat((flow.sload * 5.1).toFixed(2)),
        rate: parseFloat((flow.rate * 3.6).toFixed(2)),
        spkts: Math.round(flow.spkts * 3.2),
        smean: Math.round(flow.smean * 1.8),
      })),
  },
  {
    id: "scenario_2_categorical_shift",
    name: "Scenario 2: Protocol Frequency Skew (UDP/DNS Surge)",
    category: "categorical",
    description: "Simulated network profile transition where 65% of telemetry flows surge to UDP DNS query streams.",
    targetFeatures: ["proto", "service", "state"],
    expectedDriftLevel: "MODERATE_DRIFT",
    expectedAction: "INCREASE_MONITORING_FREQUENCY",
    operationalRationale: "Moderate categorical frequency shift without novel categories. Recommends shortening telemetry polling windows.",
    generateBatch: () =>
      createJitteredBatch(SAMPLE_BENIGN_FLOW, 35, (flow, i) => {
        if (i % 3 !== 0) {
          // 66% converted to UDP DNS flows
          return {
            ...flow,
            proto: "udp",
            service: "dns",
            state: "CON",
            dur: 0.0025,
            sbytes: 148,
            dbytes: 284,
            spkts: 2,
            dpkts: 2,
          };
        }
        return flow;
      }),
  },
  {
    id: "scenario_3_novel_categories",
    name: "Scenario 3: Zero-Day Novel Category Injection",
    category: "categorical",
    description: "Introduction of uncataloged protocols and services (e.g., WireGuard, QUIC, MQTT, CoAP) unseen in training.",
    targetFeatures: ["proto", "service"],
    expectedDriftLevel: "MODERATE_DRIFT",
    expectedAction: "RECOMMEND_DATA_COLLECTION",
    operationalRationale: "Novel protocols exceed baseline catalog. Recommends targeted data harvesting for future retraining.",
    generateBatch: () =>
      createJitteredBatch(SAMPLE_BENIGN_FLOW, 35, (flow, i) => {
        if (i % 4 === 0) {
          return { ...flow, proto: "wireguard", service: "vpn" };
        } else if (i % 4 === 1) {
          return { ...flow, proto: "quic", service: "http" };
        } else if (i % 4 === 2) {
          return { ...flow, proto: "mqtt", service: "iot" };
        }
        return flow;
      }),
  },
  {
    id: "scenario_4_data_corruption",
    name: "Scenario 4: Pipeline Data Quality Corruption",
    category: "pipeline",
    description: "Telemetry ingestion pipeline malfunction injecting missing fields and zeroed network metrics.",
    targetFeatures: ["sbytes", "dur", "sload", "dload"],
    expectedDriftLevel: "STABLE",
    expectedAction: "FLAG_AFFECTED_FEATURES",
    operationalRationale: "Data quality anomaly detected independently of statistical drift. Flags pipeline failure without blaming model.",
    generateBatch: () =>
      createJitteredBatch(SAMPLE_BENIGN_FLOW, 35, (flow, i) => {
        if (i % 3 === 0) {
          return {
            ...flow,
            sbytes: 0,
            dur: 0,
            sload: 0,
            dload: 0,
            rate: 0,
          };
        }
        return flow;
      }),
  },
  {
    id: "scenario_5_sttl_synthetic_perturbation",
    name: "Scenario 5: STTL Synthetic Artifact Perturbation",
    category: "adversarial",
    description: "Selective mutation of the Source Time-To-Live (sttl) field to evaluate C8 vulnerability vs. C10 invariance.",
    targetFeatures: ["sttl"],
    expectedDriftLevel: "MODERATE_DRIFT",
    expectedAction: "RECOMMEND_NO_STTL_FALLBACK",
    operationalRationale: "Isolated divergence on STTL testbed artifact triggers automated recommendation to activate Candidate 10.",
    generateBatch: () =>
      createJitteredBatch(SAMPLE_BENIGN_FLOW, 35, (flow) => ({
        ...flow,
        sttl: 254.0, // High synthetic artifact TTL
      })),
  },
  {
    id: "scenario_6_multivariate_shift",
    name: "Scenario 6: Combined Multivariate Network Surge",
    category: "covariate",
    description: "Simultaneous compound anomaly spanning volume, connection rates, destination concentration, and protocol mix.",
    targetFeatures: ["sbytes", "spkts", "ct_srv_dst", "rate", "service"],
    expectedDriftLevel: "CRITICAL_DRIFT",
    expectedAction: "RECOMMEND_OFFLINE_EVALUATION",
    operationalRationale: "Compound systemic shift with high prediction divergence and entropy spike. Requires SOC investigation.",
    generateBatch: () =>
      createJitteredBatch(SAMPLE_BENIGN_FLOW, 35, (flow, i) => ({
        ...flow,
        dur: parseFloat((flow.dur * 3.5).toFixed(4)),
        sbytes: Math.round(flow.sbytes * 6.2),
        spkts: Math.round(flow.spkts * 5.0),
        rate: parseFloat((flow.rate * 4.5).toFixed(2)),
        ct_srv_dst: Math.min(30, flow.ct_srv_dst + 18),
        ct_dst_src_ltm: Math.min(30, flow.ct_dst_src_ltm + 15),
        service: i % 2 === 0 ? "dns" : "smtp",
      })),
  },
];
