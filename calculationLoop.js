
import { calcFriction } from "./calcFriction.js";
import { calcDischarge } from "./calcDischarge.js";

export function calculationLoop(
    DP,
    Pd,
    Qv,
    dL,
    Div,
    e,
    dnoz,
    ksi,
    T,
    C,
    nD,
    fixedDischarge
) {
    const Pi = [];
    const Qn = [];
    const Qvn = [];
    const Qvtn = [];
    const v = [];
    const DPf = [];

    const totalLength = dL.reduce((a, b) => a + b, 0);

    for (let i = 0; i < dL.length; i++) {

        if (i === 0) {
            Pi[i] = Pd + DP;
        } else {
            const friction = calcFriction(
                dL[i],
                Div[i],
                e,
                Qvtn[i - 1],
                Pi[i - 1],
                ksi * dL[i] / totalLength,
                20
            );

            DPf[i - 1] = friction.DPf;
            v[i - 1] = friction.v;

            Pi[i] = Pi[i - 1] + DPf[i - 1];
        }

        if (fixedDischarge === 1) {
            Qvn[i] = Qv * dL[i];
        } else {
            Qvn[i] = calcDischarge(
                Pi[i] * 100,
                Pd * 100,
                dnoz[i],
                C,
                T,
                nD
            );
        }

        if (i === 0) {
            Qn[i] = 0;
            Qvtn[i] = 0;
        } else if (dnoz[i] >= 0.01) {
            Qn[i] = Qvn[i] / dL[i];
            Qvtn[i] = Qvtn[i - 1] + Qvn[i];
        } else {
            Qn[i] = 0;
            Qvtn[i] = Qvtn[i - 1];
        }
    }

    return {
        Pi,
        Qn,
        Qvtn,
        v,
        DPf
    };
}
