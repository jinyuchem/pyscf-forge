/* Copyright 2014-2026 The PySCF Developers. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

 *
 * Author: OpenAI Codex
 */

#include "config.h"
#include "np_helper/np_helper.h"
#include "vhf/fblas.h"
#include <complex.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct
{
        void *cache[6];
        short a;
        short b;
        short c;
        short _padding;
} CacheJob;

static size_t _ccsd_t_gen_jobs(CacheJob *jobs, int nocc, int nvir, int a0,
                               int a1, int b0, int b1, void *cache_row_a,
                               void *cache_col_a, void *cache_row_b,
                               void *cache_col_b, size_t stride)
{
        size_t nov = nocc * (nocc + nvir) * stride;
        int da = a1 - a0;
        int db = b1 - b0;
        size_t m, a, b, c;

        if (b1 <= a0)
        {
                m = 0;
                for (a = a0; a < a1; a++)
                {
                        for (b = b0; b < b1; b++)
                        {
                                for (c = 0; c < b0; c++, m++)
                                {
                                        jobs[m].a = a;
                                        jobs[m].b = b;
                                        jobs[m].c = c;
                                        jobs[m].cache[0] = cache_row_a + nov * (a1 * (a - a0) + b);
                                        jobs[m].cache[1] = cache_row_a + nov * (a1 * (a - a0) + c);
                                        jobs[m].cache[2] = cache_col_a + nov * (da * (b) + a - a0);
                                        jobs[m].cache[3] = cache_row_b + nov * (b1 * (b - b0) + c);
                                        jobs[m].cache[4] = cache_col_a + nov * (da * (c) + a - a0);
                                        jobs[m].cache[5] = cache_col_b + nov * (db * (c) + b - b0);
                                }
                                for (c = b0; c <= b; c++, m++)
                                {
                                        jobs[m].a = a;
                                        jobs[m].b = b;
                                        jobs[m].c = c;
                                        jobs[m].cache[0] = cache_row_a + nov * (a1 * (a - a0) + b);
                                        jobs[m].cache[1] = cache_row_a + nov * (a1 * (a - a0) + c);
                                        jobs[m].cache[2] = cache_col_a + nov * (da * (b) + a - a0);
                                        jobs[m].cache[3] = cache_row_b + nov * (b1 * (b - b0) + c);
                                        jobs[m].cache[4] = cache_col_a + nov * (da * (c) + a - a0);
                                        jobs[m].cache[5] = cache_row_b + nov * (b1 * (c - b0) + b);
                                }
                        }
                }
        }
        else
        {
                m = 0;
                for (a = a0; a < a1; a++)
                {
                        for (b = a0; b <= a; b++)
                        {
                                for (c = 0; c < a0; c++, m++)
                                {
                                        jobs[m].a = a;
                                        jobs[m].b = b;
                                        jobs[m].c = c;
                                        jobs[m].cache[0] = cache_row_a + nov * (a1 * (a - a0) + b);
                                        jobs[m].cache[1] = cache_row_a + nov * (a1 * (a - a0) + c);
                                        jobs[m].cache[2] = cache_row_a + nov * (a1 * (b - a0) + a);
                                        jobs[m].cache[3] = cache_row_a + nov * (a1 * (b - a0) + c);
                                        jobs[m].cache[4] = cache_col_a + nov * (da * (c) + a - a0);
                                        jobs[m].cache[5] = cache_col_a + nov * (da * (c) + b - a0);
                                }
                                for (c = a0; c <= b; c++, m++)
                                {
                                        jobs[m].a = a;
                                        jobs[m].b = b;
                                        jobs[m].c = c;
                                        jobs[m].cache[0] = cache_row_a + nov * (a1 * (a - a0) + b);
                                        jobs[m].cache[1] = cache_row_a + nov * (a1 * (a - a0) + c);
                                        jobs[m].cache[2] = cache_row_a + nov * (a1 * (b - a0) + a);
                                        jobs[m].cache[3] = cache_row_a + nov * (a1 * (b - a0) + c);
                                        jobs[m].cache[4] = cache_row_a + nov * (a1 * (c - a0) + a);
                                        jobs[m].cache[5] = cache_row_a + nov * (a1 * (c - a0) + b);
                                }
                        }
                }
        }
        return m;
}

static void _make_permute_indices(int *idx, int n)
{
        const int nn = n * n;
        const int nnn = nn * n;
        int *idx0 = idx;
        int *idx1 = idx0 + nnn;
        int *idx2 = idx1 + nnn;
        int *idx3 = idx2 + nnn;
        int *idx4 = idx3 + nnn;
        int *idx5 = idx4 + nnn;
        int i, j, k, m;

        for (m = 0, i = 0; i < n; i++)
        {
                for (j = 0; j < n; j++)
                {
                        for (k = 0; k < n; k++, m++)
                        {
                                idx0[m] = i * nn + j * n + k;
                                idx1[m] = i * nn + k * n + j;
                                idx2[m] = j * nn + i * n + k;
                                idx3[m] = k * nn + i * n + j;
                                idx4[m] = j * nn + k * n + i;
                                idx5[m] = k * nn + j * n + i;
                        }
                }
        }
}

/*************************************************
 *
 * UCCSD(T) same-spin part
 *
 *************************************************/

static void add_and_permute(double *out, double *w, double *v, int n)
{
        int nn = n * n;
        int nnn = nn * n;
        int i, j, k;

        for (i = 0; i < nnn; i++)
        {
                v[i] += w[i];
        }

        for (i = 0; i < n; i++)
        {
                for (j = 0; j < n; j++)
                {
                        for (k = 0; k < n; k++)
                        {
                                out[i * nn + j * n + k] =
                                    v[i * nn + j * n + k] + v[j * nn + k * n + i] +
                                    v[k * nn + i * n + j] - v[k * nn + j * n + i] -
                                    v[i * nn + k * n + j] - v[j * nn + i * n + k];
                        }
                }
        }
}

static void get_wv(double *w, double *v, double *cache, double *fvohalf,
                   double *vooo, double *vv_op, double *t1T, double *t2T,
                   int nocc, int nvir, int a, int b, int c, int *idx)
{
        const double D0 = 0;
        const double D1 = 1;
        const double DN1 = -1;
        const char TRANS_N = 'N';
        const char TRANS_T = 'T';
        const int nmo = nocc + nvir;
        const int noo = nocc * nocc;
        const size_t nooo = nocc * noo;
        const size_t nvoo = nvir * noo;
        int i, j, k, n;
        double *pt2T;

        dgemm_(&TRANS_N, &TRANS_N, &noo, &nocc, &nvir, &DN1, t2T + c * nvoo, &noo,
               vv_op + nocc, &nmo, &D0, cache, &noo);
        dgemm_(&TRANS_N, &TRANS_T, &nocc, &noo, &nocc, &DN1, t2T + b * nvoo + c * noo,
               &nocc, vooo + a * nooo, &noo, &D1, cache, &nocc);

        pt2T = t2T + a * nvoo + b * noo;
        for (n = 0, i = 0; i < nocc; i++)
        {
                for (j = 0; j < nocc; j++)
                {
                        for (k = 0; k < nocc; k++, n++)
                        {
                                w[idx[n]] += cache[n];
                                v[idx[n]] += (vv_op[i * nmo + j] * t1T[c * nocc + k] +
                                              pt2T[i * nocc + j] * fvohalf[c * nocc + k]);
                        }
                }
        }
}

static void sym_wv(double *w, double *v, double *cache, double *fvohalf,
                   double *vooo, double *vv_op, double *t1T, double *t2T,
                   int nocc, int nvir, int a, int b, int c, int nirrep,
                   int *o_ir_loc, int *v_ir_loc, int *oo_ir_loc, int *orbsym,
                   int *idx)
{
        const double D0 = 0;
        const double D1 = 1;
        const char TRANS_N = 'N';
        const int nmo = nocc + nvir;
        const int noo = nocc * nocc;
        const int nooo = nocc * noo;
        const int nvoo = nvir * noo;
        int a_irrep = orbsym[nocc + a];
        int b_irrep = orbsym[nocc + b];
        int c_irrep = orbsym[nocc + c];
        int ab_irrep = a_irrep ^ b_irrep;
        int bc_irrep = c_irrep ^ b_irrep;
        int i, j, k, n;
        int fr, f0, f1, df, mr, m0, m1, dm, mk0;
        int ir, i0, i1, di, kr, k0, k1, dk, jr;
        int ijr, ij0, ij1, dij, jkr, jk0, jk1, djk;
        double *pt2T;

        pt2T = t2T + c * nvoo;
        for (ir = 0; ir < nirrep; ir++)
        {
                i0 = o_ir_loc[ir];
                i1 = o_ir_loc[ir + 1];
                di = i1 - i0;
                if (di > 0)
                {
                        fr = ir ^ ab_irrep;
                        f0 = v_ir_loc[fr];
                        f1 = v_ir_loc[fr + 1];
                        df = f1 - f0;
                        if (df > 0)
                        {
                                jkr = fr ^ c_irrep;
                                jk0 = oo_ir_loc[jkr];
                                jk1 = oo_ir_loc[jkr + 1];
                                djk = jk1 - jk0;
                                if (djk > 0)
                                {
                                        dgemm_(&TRANS_N, &TRANS_N, &djk, &di, &df, &D1, pt2T + f0 * noo + jk0,
                                               &noo, vv_op + i0 * nmo + nocc + f0, &nmo, &D0, cache, &djk);
                                        for (n = 0, i = o_ir_loc[ir]; i < o_ir_loc[ir + 1]; i++)
                                        {
                                                for (jr = 0; jr < nirrep; jr++)
                                                {
                                                        kr = jkr ^ jr;
                                                        for (j = o_ir_loc[jr]; j < o_ir_loc[jr + 1]; j++)
                                                        {
                                                                for (k = o_ir_loc[kr]; k < o_ir_loc[kr + 1]; k++, n++)
                                                                {
                                                                        w[idx[i * noo + j * nocc + k]] -= cache[n];
                                                                }
                                                        }
                                                }
                                        }
                                }
                        }
                }
        }

        pt2T = t2T + c * nvoo + b * noo;
        vooo += a * nooo;
        mk0 = oo_ir_loc[bc_irrep];
        for (mr = 0; mr < nirrep; mr++)
        {
                m0 = o_ir_loc[mr];
                m1 = o_ir_loc[mr + 1];
                dm = m1 - m0;
                if (dm > 0)
                {
                        kr = mr ^ bc_irrep;
                        k0 = o_ir_loc[kr];
                        k1 = o_ir_loc[kr + 1];
                        dk = k1 - k0;
                        if (dk > 0)
                        {
                                ijr = mr ^ a_irrep;
                                ij0 = oo_ir_loc[ijr];
                                ij1 = oo_ir_loc[ijr + 1];
                                dij = ij1 - ij0;
                                if (dij > 0)
                                {
                                        dgemm_(&TRANS_N, &TRANS_N, &dk, &dij, &dm, &D1, pt2T + mk0, &dk,
                                               vooo + ij0 * nocc + m0, &nocc, &D0, cache, &dk);
                                        for (n = 0, ir = 0; ir < nirrep; ir++)
                                        {
                                                jr = ijr ^ ir;
                                                for (i = o_ir_loc[ir]; i < o_ir_loc[ir + 1]; i++)
                                                {
                                                        for (j = o_ir_loc[jr]; j < o_ir_loc[jr + 1]; j++)
                                                        {
                                                                for (k = o_ir_loc[kr]; k < o_ir_loc[kr + 1]; k++, n++)
                                                                {
                                                                        w[idx[i * noo + j * nocc + k]] -= cache[n];
                                                                }
                                                        }
                                                }
                                        }
                                }
                                mk0 += dm * dk;
                        }
                }
        }

        pt2T = t2T + a * nvoo + b * noo;
        for (n = 0, i = 0; i < nocc; i++)
        {
                for (j = 0; j < nocc; j++)
                {
                        for (k = 0; k < nocc; k++, n++)
                        {
                                v[idx[n]] += (vv_op[i * nmo + j] * t1T[c * nocc + k] +
                                              pt2T[i * nocc + j] * fvohalf[c * nocc + k]);
                        }
                }
        }
}

static double _get_energy_aaa_lo(double *w, double *z, double *mo_energy,
                                 double *cache, double *ulo, int nlo, int nocc,
                                 int a, int b, int c, double fac)
{
        int i, j, k, mu;
        int nn = nocc * nocc;
        double abc = mo_energy[nocc + a] + mo_energy[nocc + b] + mo_energy[nocc + c];
        double *ti = cache;
        double *zi = ti + nocc;
        double *tj = zi + nocc;
        double *zj = tj + nocc;
        double *tk = zj + nocc;
        double *zk = tk + nocc;
        double tlo, zlo;
        double denom;
        double et = 0;
        double *ulo_mu;

        for (j = 0; j < nocc; j++)
        {
                for (k = 0; k < nocc; k++)
                {
                        for (i = 0; i < nocc; i++)
                        {
                                int ijk = i * nn + j * nocc + k;
                                denom = abc - (mo_energy[i] + mo_energy[j] + mo_energy[k]);
                                ti[i] = w[ijk];
                                zi[i] = z[ijk] / denom;
                        }
                        for (mu = 0; mu < nlo; mu++)
                        {
                                ulo_mu = ulo + mu * nocc;
                                tlo = 0;
                                zlo = 0;
                                for (i = 0; i < nocc; i++)
                                {
                                        tlo += ti[i] * ulo_mu[i];
                                        zlo += zi[i] * ulo_mu[i];
                                }
                                et -= fac / 3. * tlo * zlo;
                        }
                }
        }

        for (i = 0; i < nocc; i++)
        {
                for (k = 0; k < nocc; k++)
                {
                        for (j = 0; j < nocc; j++)
                        {
                                int ijk = i * nn + j * nocc + k;
                                denom = abc - (mo_energy[i] + mo_energy[j] + mo_energy[k]);
                                tj[j] = w[ijk];
                                zj[j] = z[ijk] / denom;
                        }
                        for (mu = 0; mu < nlo; mu++)
                        {
                                ulo_mu = ulo + mu * nocc;
                                tlo = 0;
                                zlo = 0;
                                for (j = 0; j < nocc; j++)
                                {
                                        tlo += tj[j] * ulo_mu[j];
                                        zlo += zj[j] * ulo_mu[j];
                                }
                                et -= fac / 3. * tlo * zlo;
                        }
                }
        }

        for (i = 0; i < nocc; i++)
        {
                for (j = 0; j < nocc; j++)
                {
                        for (k = 0; k < nocc; k++)
                        {
                                int ijk = i * nn + j * nocc + k;
                                denom = abc - (mo_energy[i] + mo_energy[j] + mo_energy[k]);
                                tk[k] = w[ijk];
                                zk[k] = z[ijk] / denom;
                        }
                        for (mu = 0; mu < nlo; mu++)
                        {
                                ulo_mu = ulo + mu * nocc;
                                tlo = 0;
                                zlo = 0;
                                for (k = 0; k < nocc; k++)
                                {
                                        tlo += tk[k] * ulo_mu[k];
                                        zlo += zk[k] * ulo_mu[k];
                                }
                                et -= fac / 3. * tlo * zlo;
                        }
                }
        }
        return et;
}

static double contract6_aaa(int nocc, int nvir, int a, int b, int c,
                            double *mo_energy, double *t1T, double *t2T,
                            int nirrep, int *o_ir_loc, int *v_ir_loc,
                            int *oo_ir_loc, int *orbsym, double *fvo,
                            double *vooo, double *ulo, int nlo, double *cache1,
                            void **cache, int *permute_idx)
{
        int nooo = nocc * nocc * nocc;
        int *idx0 = permute_idx;
        int *idx1 = idx0 + nooo;
        int *idx2 = idx1 + nooo;
        int *idx3 = idx2 + nooo;
        int *idx4 = idx3 + nooo;
        int *idx5 = idx4 + nooo;
        double *v0 = cache1;
        double *w0 = v0 + nooo;
        double *z0 = w0 + nooo;
        double *cache2 = z0 + nooo;
        double *wtmp = z0;
        double fac;
        int i;

        for (i = 0; i < nooo; i++)
        {
                w0[i] = 0;
                v0[i] = 0;
        }

        if (nirrep == 1)
        {
                get_wv(w0, v0, wtmp, fvo, vooo, cache[0], t1T, t2T, nocc, nvir, a, b, c,
                       idx0);
                get_wv(w0, v0, wtmp, fvo, vooo, cache[1], t1T, t2T, nocc, nvir, a, c, b,
                       idx1);
                get_wv(w0, v0, wtmp, fvo, vooo, cache[2], t1T, t2T, nocc, nvir, b, a, c,
                       idx2);
                get_wv(w0, v0, wtmp, fvo, vooo, cache[3], t1T, t2T, nocc, nvir, b, c, a,
                       idx3);
                get_wv(w0, v0, wtmp, fvo, vooo, cache[4], t1T, t2T, nocc, nvir, c, a, b,
                       idx4);
                get_wv(w0, v0, wtmp, fvo, vooo, cache[5], t1T, t2T, nocc, nvir, c, b, a,
                       idx5);
        }
        else
        {
                sym_wv(w0, v0, wtmp, fvo, vooo, cache[0], t1T, t2T, nocc, nvir, a, b, c,
                       nirrep, o_ir_loc, v_ir_loc, oo_ir_loc, orbsym, idx0);
                sym_wv(w0, v0, wtmp, fvo, vooo, cache[1], t1T, t2T, nocc, nvir, a, c, b,
                       nirrep, o_ir_loc, v_ir_loc, oo_ir_loc, orbsym, idx1);
                sym_wv(w0, v0, wtmp, fvo, vooo, cache[2], t1T, t2T, nocc, nvir, b, a, c,
                       nirrep, o_ir_loc, v_ir_loc, oo_ir_loc, orbsym, idx2);
                sym_wv(w0, v0, wtmp, fvo, vooo, cache[3], t1T, t2T, nocc, nvir, b, c, a,
                       nirrep, o_ir_loc, v_ir_loc, oo_ir_loc, orbsym, idx3);
                sym_wv(w0, v0, wtmp, fvo, vooo, cache[4], t1T, t2T, nocc, nvir, c, a, b,
                       nirrep, o_ir_loc, v_ir_loc, oo_ir_loc, orbsym, idx4);
                sym_wv(w0, v0, wtmp, fvo, vooo, cache[5], t1T, t2T, nocc, nvir, c, b, a,
                       nirrep, o_ir_loc, v_ir_loc, oo_ir_loc, orbsym, idx5);
        }
        add_and_permute(z0, w0, v0, nocc);

        if (a == c)
        {
                fac = 1. / 6;
        }
        else if (a == b || b == c)
        {
                fac = .5;
        }
        else
        {
                fac = 1.;
        }
        return _get_energy_aaa_lo(w0, z0, mo_energy, cache2, ulo, nlo, nocc, a, b, c,
                                  fac);
}

void CCulnoccsd_t_aaa(double *e_tot, double *mo_energy, double *t1T,
                      double *t2T, double *vooo, double *fvo, double *ulo,
                      int nlo, int nocc, int nvir, int a0, int a1, int b0,
                      int b1, int nirrep, int *o_ir_loc, int *v_ir_loc,
                      int *oo_ir_loc, int *orbsym, void *cache_row_a,
                      void *cache_col_a, void *cache_row_b, void *cache_col_b)
{
        int da = a1 - a0;
        int db = b1 - b0;
        CacheJob *jobs = malloc(sizeof(CacheJob) * da * db * b1);
        size_t njobs =
            _ccsd_t_gen_jobs(jobs, nocc, nvir, a0, a1, b0, b1, cache_row_a,
                             cache_col_a, cache_row_b, cache_col_b, sizeof(double));
        double *fvohalf = malloc(sizeof(double) * nvir * nocc);
        int i;
        for (i = 0; i < nvir * nocc; i++)
        {
                fvohalf[i] = fvo[i] * .5;
        }
        int *permute_idx = malloc(sizeof(int) * nocc * nocc * nocc * 6);
        _make_permute_indices(permute_idx, nocc);

#pragma omp parallel default(none)                                            \
    shared(njobs, nocc, nvir, nlo, ulo, mo_energy, t1T, t2T, vooo, fvo, jobs, \
               fvohalf, nirrep, o_ir_loc, v_ir_loc, oo_ir_loc, orbsym, e_tot, \
               permute_idx, stderr)
        {
                int a, b, c;
                size_t k;
                double *cache1 =
                    malloc(sizeof(double) * (nocc * nocc * nocc * 3 + 2 + 6 * nocc));
                double e = 0;
                if (cache1 == NULL)
                {
                        fprintf(stderr, "malloc(%zu) failed in CCulnoccsd_t_aaa\n",
                                sizeof(double) * (nocc * nocc * nocc * 3 + 6 * nocc));
                        exit(1);
                }
#pragma omp for schedule(dynamic, 4)
                for (k = 0; k < njobs; k++)
                {
                        a = jobs[k].a;
                        b = jobs[k].b;
                        c = jobs[k].c;
                        e += contract6_aaa(nocc, nvir, a, b, c, mo_energy, t1T, t2T, nirrep,
                                           o_ir_loc, v_ir_loc, oo_ir_loc, orbsym, fvohalf, vooo,
                                           ulo, nlo, cache1, jobs[k].cache, permute_idx);
                }
                free(cache1);
#pragma omp critical
                *e_tot += e;
        }
        free(fvohalf);
        free(permute_idx);
        free(jobs);
}

/*************************************************
 *
 * UCCSD(T) mixed-spin part
 *
 *************************************************/

static void get_wv_baa(double *w, double *v, double **vs_ts, double **cache,
                       int nocca, int noccb, int nvira, int nvirb, int a, int b,
                       int c, double *cache1)
{
        double *fvo = vs_ts[2];
        double *fVO = vs_ts[3];
        double *vooo = vs_ts[4];
        double *vOoO = vs_ts[5];
        double *VoOo = vs_ts[6];
        double *t1aT = vs_ts[7];
        double *t1bT = vs_ts[8];
        double *t2aaT = vs_ts[9];
        double *t2abT = vs_ts[10];
        double *vvop = cache[0];
        double *vVoP = cache[1];
        double *VvOp = cache[2];
        const double D0 = 0;
        const double D1 = 1;
        const double D2 = 2;
        const char TRANS_T = 'T';
        const char TRANS_N = 'N';
        const int nmoa = nocca + nvira;
        const int nmob = noccb + nvirb;
        const int noo = nocca * nocca;
        const int nOo = noccb * nocca;
        const size_t nooo = nocca * noo;
        const size_t noOo = nocca * nOo;
        const size_t nOoO = noccb * nOo;
        const size_t nvoo = nvira * noo;
        const int nVoO = nvirb * nOo;
        int i, j, k, n;

        dgemm_(&TRANS_T, &TRANS_T, &nocca, &nOo, &nvira, &D2, vvop + nocca, &nmoa,
               t2abT + a * nOo, &nVoO, &D0, v, &nocca);
        dgemm_(&TRANS_T, &TRANS_T, &nocca, &nOo, &nvirb, &D2, vVoP + noccb, &nmob,
               t2abT + b * (size_t)nVoO, &nOo, &D1, v, &nocca);
        dgemm_(&TRANS_N, &TRANS_T, &nOo, &nocca, &nocca, &D1, VoOo + a * noOo, &nOo,
               t2aaT + b * nvoo + c * noo, &nocca, &D1, v, &nOo);
        dgemm_(&TRANS_T, &TRANS_T, &nocca, &nOo, &noccb, &D2,
               t2abT + b * (size_t)nVoO + a * nOo, &noccb, vOoO + c * nOoO, &nOo, &D1,
               v, &nocca);
        for (n = 0, i = 0; i < noccb; i++)
        {
                for (j = 0; j < nocca; j++)
                {
                        for (k = 0; k < nocca; k++, n++)
                        {
                                w[n] = v[j * nOo + i * nocca + k];
                        }
                }
        }
        dgemm_(&TRANS_N, &TRANS_N, &noo, &noccb, &nvira, &D1, t2aaT + b * nvoo, &noo,
               VvOp + nocca, &nmoa, &D1, w, &noo);
        dgemm_(&TRANS_N, &TRANS_T, &noo, &noccb, &nocca, &D2, vooo + c * nooo, &noo,
               t2abT + b * (size_t)nVoO + a * nOo, &noccb, &D1, w, &noo);

        double *t1aT2 = cache1;
        double *fvo2 = t1aT2 + nocca;
        double *fVOhalf = fvo2 + nocca;
        for (i = 0; i < nocca; i++)
        {
                t1aT2[i] = t1aT[b * nocca + i] * 2;
                fvo2[i] = fvo[b * nocca + i] * 2;
        }
        for (i = 0; i < noccb; i++)
        {
                fVOhalf[i] = fVO[a * noccb + i] * .5;
        }
        double *pt2aaT = t2aaT + b * nvoo + c * noo;
        double *pt2abT = t2abT + (c * nvirb + a) * nOo;
        for (n = 0, i = 0; i < noccb; i++)
        {
                for (j = 0; j < nocca; j++)
                {
                        for (k = 0; k < nocca; k++, n++)
                        {
                                v[n] = (w[n] + vvop[k * nmoa + j] * t1bT[a * noccb + i] +
                                        VvOp[i * nmoa + k] * t1aT2[j] +
                                        pt2aaT[j * nocca + k] * fVOhalf[i] +
                                        pt2abT[k * noccb + i] * fvo2[j]);
                        }
                }
        }
}

static void permute_baa(double *out, double *w, int nocca, int noccb)
{
        int noo = nocca * nocca;
        int n;
        int i, j, k;

        for (n = 0, i = 0; i < noccb; i++)
        {
                for (j = 0; j < nocca; j++)
                {
                        for (k = 0; k < nocca; k++, n++)
                        {
                                out[n] = w[i * noo + j * nocca + k] - w[i * noo + k * nocca + j];
                        }
                }
        }
}

static double _get_energy_baa_lo(double *z0, double *z1, double *w0, double *w1,
                                 double *mo_ea, double *mo_eb, double *uloa,
                                 int nloa, double *ulob, int nlob,
                                 double *cache, int nocca, int noccb, int a,
                                 int b, int c, double fac)
{
        int noo = nocca * nocca;
        int ib, j, k, mu;
        double abc = mo_eb[noccb + a] + mo_ea[nocca + b] + mo_ea[nocca + c];
        double *tb = cache;
        double *zb = tb + noccb;
        double *ta = zb + noccb;
        double *za = ta + nocca;
        double et = 0;
        double denom;
        double tlo, zlo;
        double *ulo_mu;

        for (j = 0; j < nocca; j++)
        {
                for (k = 0; k < nocca; k++)
                {
                        for (ib = 0; ib < noccb; ib++)
                        {
                                int idx0 = ib * noo + j * nocca + k;
                                int idx1 = ib * noo + k * nocca + j;
                                denom = abc - (mo_eb[ib] + mo_ea[j] + mo_ea[k]);
                                tb[ib] = w0[idx0] + w1[idx1];
                                zb[ib] = (z0[idx0] + z1[idx1]) / denom;
                        }
                        for (mu = 0; mu < nlob; mu++)
                        {
                                ulo_mu = ulob + mu * noccb;
                                tlo = 0;
                                zlo = 0;
                                for (ib = 0; ib < noccb; ib++)
                                {
                                        tlo += tb[ib] * ulo_mu[ib];
                                        zlo += zb[ib] * ulo_mu[ib];
                                }
                                et -= fac / 3. * tlo * zlo;
                        }
                }
        }

        for (ib = 0; ib < noccb; ib++)
        {
                for (j = 0; j < nocca; j++)
                {
                        for (k = 0; k < nocca; k++)
                        {
                                int idx0 = ib * noo + j * nocca + k;
                                int idx1 = ib * noo + k * nocca + j;
                                denom = abc - (mo_eb[ib] + mo_ea[j] + mo_ea[k]);
                                ta[k] = w0[idx0] + w1[idx1];
                                za[k] = (z0[idx0] + z1[idx1]) / denom;
                        }
                        for (mu = 0; mu < nloa; mu++)
                        {
                                ulo_mu = uloa + mu * nocca;
                                tlo = 0;
                                zlo = 0;
                                for (k = 0; k < nocca; k++)
                                {
                                        tlo += ta[k] * ulo_mu[k];
                                        zlo += za[k] * ulo_mu[k];
                                }
                                et -= fac / 3. * tlo * zlo;
                        }
                }
        }

        for (ib = 0; ib < noccb; ib++)
        {
                for (k = 0; k < nocca; k++)
                {
                        for (j = 0; j < nocca; j++)
                        {
                                int idx0 = ib * noo + j * nocca + k;
                                int idx1 = ib * noo + k * nocca + j;
                                denom = abc - (mo_eb[ib] + mo_ea[j] + mo_ea[k]);
                                ta[j] = w0[idx0] + w1[idx1];
                                za[j] = (z0[idx0] + z1[idx1]) / denom;
                        }
                        for (mu = 0; mu < nloa; mu++)
                        {
                                ulo_mu = uloa + mu * nocca;
                                tlo = 0;
                                zlo = 0;
                                for (j = 0; j < nocca; j++)
                                {
                                        tlo += ta[j] * ulo_mu[j];
                                        zlo += za[j] * ulo_mu[j];
                                }
                                et -= fac / 3. * tlo * zlo;
                        }
                }
        }
        return et;
}

static double contract6_baa(int nocca, int noccb, int nvira, int nvirb, int a,
                            int b, int c, double **vs_ts, void **cache,
                            double *cache1, double *uloa, int nloa,
                            double *ulob, int nlob)
{
        int nOoo = noccb * nocca * nocca;
        double *v0 = cache1;
        double *v1 = v0 + nOoo;
        double *w0 = v1 + nOoo;
        double *w1 = w0 + nOoo;
        double *z0 = w1 + nOoo;
        double *z1 = v0;
        double fac;
        cache1 += nOoo * 5;

        get_wv_baa(w0, v0, vs_ts, ((double **)cache), nocca, noccb, nvira, nvirb, a,
                   b, c, cache1);
        get_wv_baa(w1, v1, vs_ts, ((double **)cache) + 3, nocca, noccb, nvira, nvirb,
                   a, c, b, cache1);
        permute_baa(z0, v0, nocca, noccb);
        permute_baa(z1, v1, nocca, noccb);

        if (b == c)
        {
                fac = .5;
        }
        else
        {
                fac = 1.;
        }
        return _get_energy_baa_lo(z0, z1, w0, w1, vs_ts[0], vs_ts[1], uloa, nloa,
                                  ulob, nlob, cache1, nocca, noccb, a, b, c, fac);
}

static size_t gen_baa_jobs(CacheJob *jobs, int nocca, int noccb, int nvira,
                           int nvirb, int a0, int a1, int b0, int b1,
                           void *cache_row_a, void *cache_col_a,
                           void *cache_row_b, void *cache_col_b,
                           size_t stride)
{
        size_t nov = nocca * (nocca + nvira) * stride;
        size_t noV = nocca * (noccb + nvirb) * stride;
        size_t nOv = noccb * (nocca + nvira) * stride;
        int da = a1 - a0;
        int db = b1 - b0;
        int a, b, c;
        size_t m = 0;

        for (a = a0; a < a1; a++)
        {
                for (b = b0; b < b1; b++)
                {
                        for (c = 0; c <= b; c++, m++)
                        {
                                jobs[m].a = a;
                                jobs[m].b = b;
                                jobs[m].c = c;
                                if (c < b0)
                                {
                                        jobs[m].cache[0] = cache_col_b + nov * (db * (c) + b - b0);
                                }
                                else
                                {
                                        jobs[m].cache[0] = cache_row_b + nov * (b1 * (c - b0) + b);
                                }
                                jobs[m].cache[1] = cache_col_a + noV * (da * (c) + a - a0);
                                jobs[m].cache[2] = cache_row_a + nOv * (nvira * (a - a0) + c);
                                jobs[m].cache[3] = cache_row_b + nov * (b1 * (b - b0) + c);
                                jobs[m].cache[4] = cache_col_a + noV * (da * (b) + a - a0);
                                jobs[m].cache[5] = cache_row_a + nOv * (nvira * (a - a0) + b);
                        }
                }
        }
        return m;
}

void CCulnoccsd_t_baa(double *e_tot, double *mo_ea, double *mo_eb, double *t1aT,
                      double *t1bT, double *t2aaT, double *t2abT, double *vooo,
                      double *vOoO, double *VoOo, double *fvo, double *fVO,
                      double *uloa, int nloa, double *ulob, int nlob, int nocca,
                      int noccb, int nvira, int nvirb, int a0, int a1, int b0,
                      int b1, void *cache_row_a, void *cache_col_a,
                      void *cache_row_b, void *cache_col_b)
{
        int da = a1 - a0;
        int db = b1 - b0;
        CacheJob *jobs = malloc(sizeof(CacheJob) * da * db * b1);
        size_t njobs = gen_baa_jobs(jobs, nocca, noccb, nvira, nvirb, a0, a1, b0, b1,
                                    cache_row_a, cache_col_a, cache_row_b,
                                    cache_col_b, sizeof(double));
        double *vs_ts[] = {mo_ea, mo_eb, fvo, fVO, vooo, vOoO,
                           VoOo, t1aT, t1bT, t2aaT, t2abT};

#pragma omp parallel default(none)                                           \
    shared(njobs, nocca, noccb, nvira, nvirb, nloa, nlob, uloa, ulob, mo_ea, \
               mo_eb, vs_ts, jobs, e_tot, stderr)
        {
                int a, b, c;
                size_t k;
                double *cache1 = malloc(sizeof(double) * (noccb * nocca * nocca * 5 + 1 +
                                                          nocca * 2 + noccb * 2));
                double e = 0;
                if (cache1 == NULL)
                {
                        fprintf(stderr, "malloc failed in CCulnoccsd_t_baa\n");
                        exit(1);
                }
#pragma omp for schedule(dynamic, 4)
                for (k = 0; k < njobs; k++)
                {
                        a = jobs[k].a;
                        b = jobs[k].b;
                        c = jobs[k].c;
                        e += contract6_baa(nocca, noccb, nvira, nvirb, a, b, c, vs_ts,
                                           jobs[k].cache, cache1, uloa, nloa, ulob, nlob);
                }
                free(cache1);
#pragma omp critical
                *e_tot += e;
        }
        free(jobs);
}

/*************************************************
 *
 * Complex kernels
 *
 *************************************************/

static void zadd_and_permute(double complex *out, double complex *w,
                             double complex *v, int n)
{
        int nn = n * n;
        int nnn = nn * n;
        int i, j, k;

        for (i = 0; i < nnn; i++)
        {
                v[i] += w[i];
        }

        for (i = 0; i < n; i++)
        {
                for (j = 0; j < n; j++)
                {
                        for (k = 0; k < n; k++)
                        {
                                out[i * nn + j * n + k] =
                                    v[i * nn + j * n + k] + v[j * nn + k * n + i] +
                                    v[k * nn + i * n + j] - v[k * nn + j * n + i] -
                                    v[i * nn + k * n + j] - v[j * nn + i * n + k];
                        }
                }
        }
}

static void zget_wv(double complex *w, double complex *v, double complex *cache,
                    double complex *fvohalf, double complex *vooo,
                    double complex *vv_op, double complex *t1T,
                    double complex *t2T, int nocc, int nvir, int a, int b,
                    int c, int *idx)
{
        const double complex D0 = 0;
        const double complex D1 = 1;
        const double complex DN1 = -1;
        const char TRANS_N = 'N';
        const char TRANS_T = 'T';
        const int nmo = nocc + nvir;
        const int noo = nocc * nocc;
        const size_t nooo = nocc * noo;
        const size_t nvoo = nvir * noo;
        int i, j, k, n;
        double complex *pt2T;

        zgemm_(&TRANS_N, &TRANS_N, &noo, &nocc, &nvir, &DN1, t2T + c * nvoo, &noo,
               vv_op + nocc, &nmo, &D0, cache, &noo);
        zgemm_(&TRANS_N, &TRANS_T, &nocc, &noo, &nocc, &DN1, t2T + b * nvoo + c * noo,
               &nocc, vooo + a * nooo, &noo, &D1, cache, &nocc);

        pt2T = t2T + a * nvoo + b * noo;
        for (n = 0, i = 0; i < nocc; i++)
        {
                for (j = 0; j < nocc; j++)
                {
                        for (k = 0; k < nocc; k++, n++)
                        {
                                w[idx[n]] += cache[n];
                                v[idx[n]] += (vv_op[i * nmo + j] * t1T[c * nocc + k] +
                                              pt2T[i * nocc + j] * fvohalf[c * nocc + k]);
                        }
                }
        }
}

static double _get_energy_zaaa_lo(double complex *w, double complex *z,
                                  double *mo_energy, double complex *cache,
                                  double complex *ulo, int nlo, int nocc, int a,
                                  int b, int c, double fac)
{
        int i, j, k, mu;
        int nn = nocc * nocc;
        double abc = mo_energy[nocc + a] + mo_energy[nocc + b] + mo_energy[nocc + c];
        double complex *ti = cache;
        double complex *zi = ti + nocc;
        double complex *tj = zi + nocc;
        double complex *zj = tj + nocc;
        double complex *tk = zj + nocc;
        double complex *zk = tk + nocc;
        double complex tlo, zlo;
        double denom;
        double et = 0;
        double complex *ulo_mu;

        for (j = 0; j < nocc; j++)
        {
                for (k = 0; k < nocc; k++)
                {
                        for (i = 0; i < nocc; i++)
                        {
                                int ijk = i * nn + j * nocc + k;
                                denom = abc - (mo_energy[i] + mo_energy[j] + mo_energy[k]);
                                ti[i] = w[ijk];
                                zi[i] = z[ijk] / denom;
                        }
                        for (mu = 0; mu < nlo; mu++)
                        {
                                ulo_mu = ulo + mu * nocc;
                                tlo = 0;
                                zlo = 0;
                                for (i = 0; i < nocc; i++)
                                {
                                        tlo += ti[i] * conj(ulo_mu[i]);
                                        zlo += zi[i] * conj(ulo_mu[i]);
                                }
                                et -= fac / 3. * creal(tlo * conj(zlo));
                        }
                }
        }

        for (i = 0; i < nocc; i++)
        {
                for (k = 0; k < nocc; k++)
                {
                        for (j = 0; j < nocc; j++)
                        {
                                int ijk = i * nn + j * nocc + k;
                                denom = abc - (mo_energy[i] + mo_energy[j] + mo_energy[k]);
                                tj[j] = w[ijk];
                                zj[j] = z[ijk] / denom;
                        }
                        for (mu = 0; mu < nlo; mu++)
                        {
                                ulo_mu = ulo + mu * nocc;
                                tlo = 0;
                                zlo = 0;
                                for (j = 0; j < nocc; j++)
                                {
                                        tlo += tj[j] * conj(ulo_mu[j]);
                                        zlo += zj[j] * conj(ulo_mu[j]);
                                }
                                et -= fac / 3. * creal(tlo * conj(zlo));
                        }
                }
        }

        for (i = 0; i < nocc; i++)
        {
                for (j = 0; j < nocc; j++)
                {
                        for (k = 0; k < nocc; k++)
                        {
                                int ijk = i * nn + j * nocc + k;
                                denom = abc - (mo_energy[i] + mo_energy[j] + mo_energy[k]);
                                tk[k] = w[ijk];
                                zk[k] = z[ijk] / denom;
                        }
                        for (mu = 0; mu < nlo; mu++)
                        {
                                ulo_mu = ulo + mu * nocc;
                                tlo = 0;
                                zlo = 0;
                                for (k = 0; k < nocc; k++)
                                {
                                        tlo += tk[k] * conj(ulo_mu[k]);
                                        zlo += zk[k] * conj(ulo_mu[k]);
                                }
                                et -= fac / 3. * creal(tlo * conj(zlo));
                        }
                }
        }
        return et;
}

static double zcontract6_aaa(int nocc, int nvir, int a, int b, int c,
                             double *mo_energy, double complex *t1T,
                             double complex *t2T, double complex *ulo, int nlo,
                             double complex *fvo, double complex *vooo,
                             double complex *cache1, void **cache,
                             int *permute_idx)
{
        int nooo = nocc * nocc * nocc;
        int *idx0 = permute_idx;
        int *idx1 = idx0 + nooo;
        int *idx2 = idx1 + nooo;
        int *idx3 = idx2 + nooo;
        int *idx4 = idx3 + nooo;
        int *idx5 = idx4 + nooo;
        double complex *v0 = cache1;
        double complex *w0 = v0 + nooo;
        double complex *z0 = w0 + nooo;
        double complex *cache2 = z0 + nooo;
        double complex *wtmp = z0;
        double fac;
        int i;

        for (i = 0; i < nooo; i++)
        {
                w0[i] = 0;
                v0[i] = 0;
        }

        zget_wv(w0, v0, wtmp, fvo, vooo, cache[0], t1T, t2T, nocc, nvir, a, b, c,
                idx0);
        zget_wv(w0, v0, wtmp, fvo, vooo, cache[1], t1T, t2T, nocc, nvir, a, c, b,
                idx1);
        zget_wv(w0, v0, wtmp, fvo, vooo, cache[2], t1T, t2T, nocc, nvir, b, a, c,
                idx2);
        zget_wv(w0, v0, wtmp, fvo, vooo, cache[3], t1T, t2T, nocc, nvir, b, c, a,
                idx3);
        zget_wv(w0, v0, wtmp, fvo, vooo, cache[4], t1T, t2T, nocc, nvir, c, a, b,
                idx4);
        zget_wv(w0, v0, wtmp, fvo, vooo, cache[5], t1T, t2T, nocc, nvir, c, b, a,
                idx5);
        zadd_and_permute(z0, w0, v0, nocc);

        if (a == c)
        {
                fac = 1. / 6;
        }
        else if (a == b || b == c)
        {
                fac = .5;
        }
        else
        {
                fac = 1.;
        }
        return _get_energy_zaaa_lo(w0, z0, mo_energy, cache2, ulo, nlo, nocc, a, b, c,
                                   fac);
}

void CCulnoccsd_t_zaaa(double complex *e_tot, double *mo_energy,
                       double complex *t1T, double complex *t2T,
                       double complex *vooo, double complex *fvo,
                       double complex *ulo, int nlo, int nocc, int nvir, int a0,
                       int a1, int b0, int b1, int nirrep, int *o_ir_loc,
                       int *v_ir_loc, int *oo_ir_loc, int *orbsym,
                       void *cache_row_a, void *cache_col_a, void *cache_row_b,
                       void *cache_col_b)
{
        int da = a1 - a0;
        int db = b1 - b0;
        CacheJob *jobs = malloc(sizeof(CacheJob) * da * db * b1);
        size_t njobs = _ccsd_t_gen_jobs(jobs, nocc, nvir, a0, a1, b0, b1, cache_row_a,
                                        cache_col_a, cache_row_b, cache_col_b,
                                        sizeof(double complex));
        double complex *fvohalf = malloc(sizeof(double complex) * nvir * nocc);
        int i;
        for (i = 0; i < nvir * nocc; i++)
        {
                fvohalf[i] = fvo[i] * .5;
        }
        int *permute_idx = malloc(sizeof(int) * nocc * nocc * nocc * 6);
        _make_permute_indices(permute_idx, nocc);
        (void)nirrep;
        (void)o_ir_loc;
        (void)v_ir_loc;
        (void)oo_ir_loc;
        (void)orbsym;

#pragma omp parallel default(none)                                          \
    shared(njobs, nocc, nvir, nlo, ulo, mo_energy, t1T, t2T, vooo, fvohalf, \
               jobs, e_tot, permute_idx, stderr)
        {
                int a, b, c;
                size_t k;
                double complex *cache1 = malloc(sizeof(double complex) *
                                                (nocc * nocc * nocc * 3 + 2 + 6 * nocc));
                double e = 0;
                if (cache1 == NULL)
                {
                        fprintf(stderr, "malloc(%zu) failed in CCulnoccsd_t_zaaa\n",
                                sizeof(double complex) * (nocc * nocc * nocc * 3 + 6 * nocc));
                        exit(1);
                }
#pragma omp for schedule(dynamic, 4)
                for (k = 0; k < njobs; k++)
                {
                        a = jobs[k].a;
                        b = jobs[k].b;
                        c = jobs[k].c;
                        e += zcontract6_aaa(nocc, nvir, a, b, c, mo_energy, t1T, t2T, ulo, nlo,
                                            fvohalf, vooo, cache1, jobs[k].cache, permute_idx);
                }
                free(cache1);
#pragma omp critical
                *e_tot += e;
        }
        free(fvohalf);
        free(permute_idx);
        free(jobs);
}

static void zget_wv_baa(double complex *w, double complex *v,
                        double complex **vs_ts, double complex **cache,
                        int nocca, int noccb, int nvira, int nvirb, int a,
                        int b, int c, double complex *cache1)
{
        double complex *fvo = vs_ts[2];
        double complex *fVO = vs_ts[3];
        double complex *vooo = vs_ts[4];
        double complex *vOoO = vs_ts[5];
        double complex *VoOo = vs_ts[6];
        double complex *t1aT = vs_ts[7];
        double complex *t1bT = vs_ts[8];
        double complex *t2aaT = vs_ts[9];
        double complex *t2abT = vs_ts[10];
        double complex *vvop = cache[0];
        double complex *vVoP = cache[1];
        double complex *VvOp = cache[2];
        const double complex D0 = 0;
        const double complex D1 = 1;
        const double complex D2 = 2;
        const char TRANS_T = 'T';
        const char TRANS_N = 'N';
        const int nmoa = nocca + nvira;
        const int nmob = noccb + nvirb;
        const int noo = nocca * nocca;
        const int nOo = noccb * nocca;
        const size_t nooo = nocca * noo;
        const size_t noOo = nocca * nOo;
        const size_t nOoO = noccb * nOo;
        const size_t nvoo = nvira * noo;
        const int nVoO = nvirb * nOo;
        int i, j, k, n;

        zgemm_(&TRANS_T, &TRANS_T, &nocca, &nOo, &nvira, &D2, vvop + nocca, &nmoa,
               t2abT + a * nOo, &nVoO, &D0, v, &nocca);
        zgemm_(&TRANS_T, &TRANS_T, &nocca, &nOo, &nvirb, &D2, vVoP + noccb, &nmob,
               t2abT + b * (size_t)nVoO, &nOo, &D1, v, &nocca);
        zgemm_(&TRANS_N, &TRANS_T, &nOo, &nocca, &nocca, &D1, VoOo + a * noOo, &nOo,
               t2aaT + b * nvoo + c * noo, &nocca, &D1, v, &nOo);
        zgemm_(&TRANS_T, &TRANS_T, &nocca, &nOo, &noccb, &D2,
               t2abT + b * (size_t)nVoO + a * nOo, &noccb, vOoO + c * nOoO, &nOo, &D1,
               v, &nocca);
        for (n = 0, i = 0; i < noccb; i++)
        {
                for (j = 0; j < nocca; j++)
                {
                        for (k = 0; k < nocca; k++, n++)
                        {
                                w[n] = v[j * nOo + i * nocca + k];
                        }
                }
        }
        zgemm_(&TRANS_N, &TRANS_N, &noo, &noccb, &nvira, &D1, t2aaT + b * nvoo, &noo,
               VvOp + nocca, &nmoa, &D1, w, &noo);
        zgemm_(&TRANS_N, &TRANS_T, &noo, &noccb, &nocca, &D2, vooo + c * nooo, &noo,
               t2abT + b * (size_t)nVoO + a * nOo, &noccb, &D1, w, &noo);

        double complex *t1aT2 = cache1;
        double complex *fvo2 = t1aT2 + nocca;
        double complex *fVOhalf = fvo2 + nocca;
        for (i = 0; i < nocca; i++)
        {
                t1aT2[i] = t1aT[b * nocca + i] * 2;
                fvo2[i] = fvo[b * nocca + i] * 2;
        }
        for (i = 0; i < noccb; i++)
        {
                fVOhalf[i] = fVO[a * noccb + i] * .5;
        }
        double complex *pt2aaT = t2aaT + b * nvoo + c * noo;
        double complex *pt2abT = t2abT + (c * nvirb + a) * nOo;
        for (n = 0, i = 0; i < noccb; i++)
        {
                for (j = 0; j < nocca; j++)
                {
                        for (k = 0; k < nocca; k++, n++)
                        {
                                v[n] = (w[n] + vvop[k * nmoa + j] * t1bT[a * noccb + i] +
                                        VvOp[i * nmoa + k] * t1aT2[j] +
                                        pt2aaT[j * nocca + k] * fVOhalf[i] +
                                        pt2abT[k * noccb + i] * fvo2[j]);
                        }
                }
        }
}

static void zpermute_baa(double complex *out, double complex *w, int nocca,
                         int noccb)
{
        int noo = nocca * nocca;
        int n;
        int i, j, k;

        for (n = 0, i = 0; i < noccb; i++)
        {
                for (j = 0; j < nocca; j++)
                {
                        for (k = 0; k < nocca; k++, n++)
                        {
                                out[n] = w[i * noo + j * nocca + k] - w[i * noo + k * nocca + j];
                        }
                }
        }
}

static double _get_energy_zbaa_lo(double complex *z0, double complex *z1,
                                  double complex *w0, double complex *w1,
                                  double *mo_ea, double *mo_eb,
                                  double complex *uloa, int nloa,
                                  double complex *ulob, int nlob,
                                  double complex *cache, int nocca, int noccb,
                                  int a, int b, int c, double fac)
{
        int noo = nocca * nocca;
        int ib, j, k, mu;
        double abc = mo_eb[noccb + a] + mo_ea[nocca + b] + mo_ea[nocca + c];
        double complex *tb = cache;
        double complex *zb = tb + noccb;
        double complex *ta = zb + noccb;
        double complex *za = ta + nocca;
        double et = 0;
        double denom;
        double complex tlo, zlo;
        double complex *ulo_mu;

        for (j = 0; j < nocca; j++)
        {
                for (k = 0; k < nocca; k++)
                {
                        for (ib = 0; ib < noccb; ib++)
                        {
                                int idx0 = ib * noo + j * nocca + k;
                                int idx1 = ib * noo + k * nocca + j;
                                denom = abc - (mo_eb[ib] + mo_ea[j] + mo_ea[k]);
                                tb[ib] = w0[idx0] + w1[idx1];
                                zb[ib] = (z0[idx0] + z1[idx1]) / denom;
                        }
                        for (mu = 0; mu < nlob; mu++)
                        {
                                ulo_mu = ulob + mu * noccb;
                                tlo = 0;
                                zlo = 0;
                                for (ib = 0; ib < noccb; ib++)
                                {
                                        tlo += tb[ib] * conj(ulo_mu[ib]);
                                        zlo += zb[ib] * conj(ulo_mu[ib]);
                                }
                                et -= fac / 3. * creal(tlo * conj(zlo));
                        }
                }
        }

        for (ib = 0; ib < noccb; ib++)
        {
                for (j = 0; j < nocca; j++)
                {
                        for (k = 0; k < nocca; k++)
                        {
                                int idx0 = ib * noo + j * nocca + k;
                                int idx1 = ib * noo + k * nocca + j;
                                denom = abc - (mo_eb[ib] + mo_ea[j] + mo_ea[k]);
                                ta[k] = w0[idx0] + w1[idx1];
                                za[k] = (z0[idx0] + z1[idx1]) / denom;
                        }
                        for (mu = 0; mu < nloa; mu++)
                        {
                                ulo_mu = uloa + mu * nocca;
                                tlo = 0;
                                zlo = 0;
                                for (k = 0; k < nocca; k++)
                                {
                                        tlo += ta[k] * conj(ulo_mu[k]);
                                        zlo += za[k] * conj(ulo_mu[k]);
                                }
                                et -= fac / 3. * creal(tlo * conj(zlo));
                        }
                }
        }

        for (ib = 0; ib < noccb; ib++)
        {
                for (k = 0; k < nocca; k++)
                {
                        for (j = 0; j < nocca; j++)
                        {
                                int idx0 = ib * noo + j * nocca + k;
                                int idx1 = ib * noo + k * nocca + j;
                                denom = abc - (mo_eb[ib] + mo_ea[j] + mo_ea[k]);
                                ta[j] = w0[idx0] + w1[idx1];
                                za[j] = (z0[idx0] + z1[idx1]) / denom;
                        }
                        for (mu = 0; mu < nloa; mu++)
                        {
                                ulo_mu = uloa + mu * nocca;
                                tlo = 0;
                                zlo = 0;
                                for (j = 0; j < nocca; j++)
                                {
                                        tlo += ta[j] * conj(ulo_mu[j]);
                                        zlo += za[j] * conj(ulo_mu[j]);
                                }
                                et -= fac / 3. * creal(tlo * conj(zlo));
                        }
                }
        }
        return et;
}

static double zcontract6_baa(int nocca, int noccb, int nvira, int nvirb, int a,
                             int b, int c, double complex **vs_ts, void **cache,
                             double complex *cache1, double complex *uloa,
                             int nloa, double complex *ulob, int nlob)
{
        int nOoo = noccb * nocca * nocca;
        double complex *v0 = cache1;
        double complex *v1 = v0 + nOoo;
        double complex *w0 = v1 + nOoo;
        double complex *w1 = w0 + nOoo;
        double complex *z0 = w1 + nOoo;
        double complex *z1 = v0;
        double fac;
        cache1 += nOoo * 5;

        zget_wv_baa(w0, v0, vs_ts, ((double complex **)cache), nocca, noccb, nvira,
                    nvirb, a, b, c, cache1);
        zget_wv_baa(w1, v1, vs_ts, ((double complex **)cache) + 3, nocca, noccb,
                    nvira, nvirb, a, c, b, cache1);
        zpermute_baa(z0, v0, nocca, noccb);
        zpermute_baa(z1, v1, nocca, noccb);

        if (b == c)
        {
                fac = .5;
        }
        else
        {
                fac = 1.;
        }
        double *mo_ea = (double *)vs_ts[0];
        double *mo_eb = (double *)vs_ts[1];
        return _get_energy_zbaa_lo(z0, z1, w0, w1, mo_ea, mo_eb, uloa, nloa, ulob,
                                   nlob, cache1, nocca, noccb, a, b, c, fac);
}

void CCulnoccsd_t_zbaa(double complex *e_tot, double *mo_ea, double *mo_eb,
                       double complex *t1aT, double complex *t1bT,
                       double complex *t2aaT, double complex *t2abT,
                       double complex *vooo, double complex *vOoO,
                       double complex *VoOo, double complex *fvo,
                       double complex *fVO, double complex *uloa, int nloa,
                       double complex *ulob, int nlob, int nocca, int noccb,
                       int nvira, int nvirb, int a0, int a1, int b0, int b1,
                       void *cache_row_a, void *cache_col_a, void *cache_row_b,
                       void *cache_col_b)
{
        int da = a1 - a0;
        int db = b1 - b0;
        CacheJob *jobs = malloc(sizeof(CacheJob) * da * db * b1);
        size_t njobs = gen_baa_jobs(jobs, nocca, noccb, nvira, nvirb, a0, a1, b0, b1,
                                    cache_row_a, cache_col_a, cache_row_b,
                                    cache_col_b, sizeof(double complex));
        double complex *vs_ts[] = {(double complex *)mo_ea, (double complex *)mo_eb,
                                   fvo, fVO, vooo, vOoO, VoOo, t1aT, t1bT, t2aaT,
                                   t2abT};

#pragma omp parallel default(none)                                           \
    shared(njobs, nocca, noccb, nvira, nvirb, nloa, nlob, uloa, ulob, mo_ea, \
               mo_eb, vs_ts, jobs, e_tot, stderr)
        {
                int a, b, c;
                size_t k;
                double complex *cache1 =
                    malloc(sizeof(double complex) *
                           (noccb * nocca * nocca * 5 + 1 + nocca * 2 + noccb * 2));
                double e = 0;
                if (cache1 == NULL)
                {
                        fprintf(stderr, "malloc failed in CCulnoccsd_t_zbaa\n");
                        exit(1);
                }
#pragma omp for schedule(dynamic, 4)
                for (k = 0; k < njobs; k++)
                {
                        a = jobs[k].a;
                        b = jobs[k].b;
                        c = jobs[k].c;
                        e += zcontract6_baa(nocca, noccb, nvira, nvirb, a, b, c, vs_ts,
                                            jobs[k].cache, cache1, uloa, nloa, ulob, nlob);
                }
                free(cache1);
#pragma omp critical
                *e_tot += e;
        }
        free(jobs);
}
