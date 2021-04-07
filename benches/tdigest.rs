use criterion::{criterion_group, criterion_main, BatchSize, Criterion, Throughput};
use fast_quantile_sketch::{sort_unstable_f32, total_cmp_f32, TDigest};
use rand::{distributions::Distribution, rngs::SmallRng, SeedableRng};
use statrs::distribution::Exponential;

fn small_rng() -> SmallRng {
    SmallRng::seed_from_u64(0xdeadb33f)
}

fn sample_distr(rng: &mut SmallRng, distr: impl Distribution<f64>, num_samples: usize) -> Vec<f32> {
    distr
        .sample_iter(rng)
        .take(num_samples)
        .map(|x| x as f32)
        .collect::<Vec<_>>()
}

fn sort_unstable_f32_partial_ord(values: &mut [f32]) {
    values.sort_by(|a, b| a.partial_cmp(b).unwrap())
}

fn sort_stable_f32(values: &mut [f32]) {
    values.sort_by(total_cmp_f32)
}

fn exp_digest(rng: &mut SmallRng, digest_size: usize, num_samples: usize) -> TDigest {
    let mut samples = sample_distr(rng, Exponential::new(1.0).unwrap(), num_samples);
    let mut digest = TDigest::new_with_size(digest_size);
    digest.merge_unsorted(&mut samples);
    digest
}

fn sort_f32_bench(c: &mut Criterion) {
    const NUM_SAMPLES: usize = 1000;

    let mut rng = small_rng();
    let samples = sample_distr(&mut rng, Exponential::new(1.0).unwrap(), NUM_SAMPLES);

    let mut g = c.benchmark_group("sort_f32");

    g.throughput(Throughput::Elements(NUM_SAMPLES as u64));
    g.bench_function("unstable[total_cmp]", |b| {
        b.iter_batched(
            || samples.clone(),
            |mut samples| sort_unstable_f32(&mut samples),
            BatchSize::SmallInput,
        )
    });
    g.bench_function("stable[total_cmp]", |b| {
        b.iter_batched(
            || samples.clone(),
            |mut samples| sort_stable_f32(&mut samples),
            BatchSize::SmallInput,
        )
    });
    g.bench_function("unstable[partial_cmp]", |b| {
        b.iter_batched(
            || samples.clone(),
            |mut samples| sort_unstable_f32_partial_ord(&mut samples),
            BatchSize::SmallInput,
        )
    });
    g.finish();
}

fn merge_buffer_fresh_bench(c: &mut Criterion) {
    const NUM_SAMPLES: usize = 1000;
    const DIGEST_SIZE: usize = 100;

    let mut rng = small_rng();
    let mut samples = sample_distr(&mut rng, Exponential::new(1.0).unwrap(), NUM_SAMPLES);
    sort_unstable_f32(&mut samples);

    let mut g = c.benchmark_group("merge_buffer_fresh");
    g.throughput(Throughput::Elements(NUM_SAMPLES as u64));
    g.bench_function("merge_sorted", |b| {
        b.iter(|| {
            let mut digest = TDigest::new_with_size(DIGEST_SIZE);
            digest.merge_sorted(&samples);
        })
    });
    g.bench_function("merge_v2", |b| {
        b.iter(|| {
            let mut digest = TDigest::new_with_size(DIGEST_SIZE);
            digest.merge_v2(samples.as_slice());
        })
    });
    g.finish();
}

fn merge_buffer_full_bench(c: &mut Criterion) {
    const NUM_SAMPLES: usize = 1000;
    const DIGEST_SIZE: usize = 100;

    let mut rng = small_rng();

    let digest1 = exp_digest(&mut rng, DIGEST_SIZE, NUM_SAMPLES);
    let mut samples = sample_distr(&mut rng, Exponential::new(1.0).unwrap(), NUM_SAMPLES);
    sort_unstable_f32(&mut samples);

    let mut g = c.benchmark_group("merge_buffer_full");
    g.throughput(Throughput::Elements(NUM_SAMPLES as u64));
    g.bench_function("merge_sorted", |b| {
        b.iter_batched(
            || digest1.clone(),
            |mut digest1| digest1.merge_sorted(&samples),
            BatchSize::SmallInput,
        )
    });
    g.bench_function("merge_v2", |b| {
        b.iter_batched(
            || digest1.clone(),
            |mut digest1| digest1.merge_v2(samples.as_slice()),
            BatchSize::SmallInput,
        )
    });
    g.finish();
}

fn merge_digest_full_bench(c: &mut Criterion) {
    const NUM_SAMPLES: usize = 1000;
    const DIGEST_SIZE: usize = 100;

    let mut rng = small_rng();

    let digest1 = exp_digest(&mut rng, DIGEST_SIZE, NUM_SAMPLES);
    let digest2 = exp_digest(&mut rng, DIGEST_SIZE, NUM_SAMPLES);

    let mut g = c.benchmark_group("merge_digest_full");
    g.bench_function("merge_v2", |b| {
        b.iter_batched(
            || (digest1.clone(), digest2.clone()),
            |(mut digest1, digest2)| digest1.merge_v2(digest2),
            BatchSize::SmallInput,
        )
    });
    g.finish();
}

fn quantile_bench(c: &mut Criterion) {
    const NUM_SAMPLES: usize = 1000;
    const DIGEST_SIZE: usize = 100;
    const QUANTILES: &[f32] = &[0.50, 0.90, 0.99, 0.999, 1.0];

    let mut rng = small_rng();
    let digest = exp_digest(&mut rng, DIGEST_SIZE, NUM_SAMPLES);

    let mut g = c.benchmark_group("quantile");
    g.throughput(Throughput::Elements(QUANTILES.len() as u64));
    g.bench_function("quantile_v1", |b| {
        b.iter(|| {
            for q in QUANTILES {
                digest.quantile(*q);
            }
        })
    });
    g.bench_function("quantile_v2", |b| {
        b.iter(|| {
            for q in QUANTILES {
                digest.quantile_v2(*q);
            }
        })
    });
    g.bench_function("quantile_v3", |b| {
        b.iter(|| {
            for q in QUANTILES {
                digest.quantile_v3(*q);
            }
        })
    });
    g.finish();
}

criterion_group!(
    tdigest_benches,
    sort_f32_bench,
    merge_buffer_fresh_bench,
    merge_buffer_full_bench,
    merge_digest_full_bench,
    quantile_bench,
);
criterion_main!(tdigest_benches);
