use std::{cmp::Ordering, iter, mem, slice, vec};

/// Maps a quantile `q \in [0, 1]` into k-weight space. The derivative of `k` is
/// inversely proportional to the number of samples that can fit into a centroid
/// at that quantile. This `k` function allows more samples per centroid near the
/// median (q = 0.5) and less near the tails (q = 0.0, q = 1.0), which increases
/// the accuracy of quantile estimates near the tails at the expense of accuracy
/// closer to the median.
///
/// The `d` parameter controls the "compression" of the T-Digest. In a fully
/// merged T-Digest, the number of centroids, `|C|` will always be in the range
/// `[floor(d/2), ceil(d)]`.
///
/// We don't actually use the `k` function itself; instead, we use its inverse,
/// `k_inv`. This definition is left here for readability.
///
/// ```ignore
/// k(q, d) if q >= 1/2 := d (1 - sqrt(2(1-q)) / 2)
/// k(q, d) if q < 1/2  := d sqrt(2q) / 2
///
/// k(0, d) = 0
/// k(1, d) = d
///
/// k'(q, d) if q >= 1/2 := d / (2 sqrt(2(1-q)))
/// k'(q, d) if q < 1/2  := d / (2 sqrt(2q))
///
/// lim_{q -> 1-} (k'(q, d)) = inf
/// lim_{q -> 0+} (k'(q, d)) = inf
/// ```
#[allow(dead_code)]
fn k(q: f32, d: f32) -> f32 {
    if q >= 0.5 {
        d - d * (0.5 - 0.5 * q).sqrt()
    } else {
        d * (0.5 * q).sqrt()
    }
}

/// The inverse of `k`. It maps k-weights to their corresponding quantiles.
///
/// ```ignore
/// k^-1(k, d) if k/d >= 1/2 := 1 - 2 (1 - k/d)^2
/// k^-1(k, d) if k/d < 1/2  := 2 (k/d)^2
/// ```
fn k_inv(k: f32, d: f32) -> f32 {
    let a = k / d;
    if a >= 0.5 {
        let b = 1.0 - a;
        1.0 - 2.0 * b * b
    } else {
        2.0 * a * a
    }
}

/// Provides a total ordering over f32's.
// TODO(philiphayes): use `std::f32::total_cmp` when it stabilizes.
#[inline]
pub fn total_cmp_f32(a: &f32, b: &f32) -> Ordering {
    let mut a = a.to_bits() as i32;
    let mut b = b.to_bits() as i32;

    a ^= (((a >> 31) as u32) >> 1) as i32;
    b ^= (((b >> 31) as u32) >> 1) as i32;

    a.cmp(&b)
}

/// Returns `true` if the iterator `iter` is sorted, according to the comparator
/// function `compare`, i.e., `x_1 <= x2 <= ... <= x_n`.
// TODO(philiphayes): use `std::slice::is_sorted_by` when it stabilizes.
fn is_sorted_by<T, F>(mut iter: impl Iterator<Item = T>, mut compare: F) -> bool
where
    F: FnMut(&T, &T) -> Option<Ordering>,
{
    let mut prev = match iter.next() {
        Some(first) => first,
        None => return true,
    };

    for next in iter {
        if let Some(Ordering::Greater) | None = compare(&prev, &next) {
            return false;
        }
        prev = next;
    }

    true
}

/// Returns `true` if the iterator of `f32`'s is totally ordered (according to
/// `total_cmp_f32`).
fn is_sorted_f32(iter: impl Iterator<Item = f32>) -> bool {
    is_sorted_by(iter, |a, b| Some(total_cmp_f32(a, b)))
}

/// Sort the slice of `f32`'s according to the total ordering given by
/// `total_cmp_f32`.
pub fn sort_unstable_f32(values: &mut [f32]) {
    values.sort_unstable_by(total_cmp_f32)
}

// TODO(philiphayes): use `std::cmp::min_by` when it stabilizes.
#[inline]
fn min_by<T, F>(a: T, b: T, compare: F) -> T
where
    F: FnOnce(&T, &T) -> Ordering,
{
    match compare(&a, &b) {
        Ordering::Less | Ordering::Equal => a,
        Ordering::Greater => b,
    }
}

// TODO(philiphayes): use `std::cmp::max_by` when it stabilizes.
#[inline]
fn max_by<T, F>(a: T, b: T, compare: F) -> T
where
    F: FnOnce(&T, &T) -> Ordering,
{
    match compare(&a, &b) {
        Ordering::Less | Ordering::Equal => b,
        Ordering::Greater => a,
    }
}

#[inline]
fn min_f32(a: f32, b: f32) -> f32 {
    min_by(a, b, total_cmp_f32)
}

#[inline]
fn max_f32(a: f32, b: f32) -> f32 {
    max_by(a, b, total_cmp_f32)
}

#[inline]
fn clamp_f32(x: f32, min: f32, max: f32) -> f32 {
    debug_assert!(min <= max);
    if x > max {
        max
    } else if x < min {
        min
    } else {
        x
    }
}

/// Linearly interpolate between `x0` and `x1` using `t \in [0, 1]`. Returns `x0`
/// when `t = 0` and `x1` when `t = 1`.
#[inline]
fn lerp_f32(x0: f32, x1: f32, t: f32) -> f32 {
    debug_assert!((0.0..=1.0).contains(&t));
    (1.0 - t) * x0 + t * x1
}

#[inline]
fn lerp_clamp_f32(x0: f32, x1: f32, t: f32) -> f32 {
    clamp_f32(lerp_f32(x0, x1, t), x0, x1)
}

// TODO(philiphayes): could store the prefix_count here? then we could binary
// search when computing the quantiles.
#[derive(Clone, Copy, Debug)]
pub struct Centroid {
    mean: f32,
    /// use float here to avoid lots of usize <-> float conversions
    count: f32,
}

impl Centroid {
    #[inline]
    fn unit(mean: f32) -> Self {
        Self { mean, count: 1.0 }
    }

    #[inline]
    fn is_unit(&self) -> bool {
        self.count <= 1.0
    }

    fn add(&mut self, sum: f32, count: f32) -> f32 {
        let new_sum = sum + (self.count * self.mean);
        let new_count = count + self.count;
        self.count = new_count;
        self.mean = new_sum / new_count;
        new_sum
    }
}

impl PartialOrd for Centroid {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(total_cmp_f32(&self.mean, &other.mean))
    }
}

impl PartialEq for Centroid {
    fn eq(&self, other: &Self) -> bool {
        matches!(total_cmp_f32(&self.mean, &other.mean), Ordering::Equal)
    }
}

pub trait Digest {
    type CentroidIter: Iterator<Item = Centroid>;

    fn centroids(self) -> Self::CentroidIter;
    fn is_empty(&self) -> bool;
    fn count(&self) -> f32;
    fn min(&self) -> f32;
    fn max(&self) -> f32;
}

// TODO(philiphayes): temporary hack that assumes the slice is always sorted; weight
// should use a newtype wrapper or something to prove sorted-ness.
impl<'a> Digest for &'a [f32] {
    type CentroidIter = iter::Map<slice::Iter<'a, f32>, fn(&f32) -> Centroid>;

    fn centroids(self) -> Self::CentroidIter {
        self.iter().map(|x| Centroid::unit(*x))
    }

    fn is_empty(&self) -> bool {
        <[f32]>::is_empty(self)
    }

    fn count(&self) -> f32 {
        self.len() as f32
    }

    fn min(&self) -> f32 {
        self[0]
    }

    fn max(&self) -> f32 {
        self[self.len() - 1]
    }
}

impl Digest for TDigest {
    type CentroidIter = vec::IntoIter<Centroid>;

    fn centroids(self) -> Self::CentroidIter {
        self.centroids.into_iter()
    }

    fn is_empty(&self) -> bool {
        self.centroids.is_empty()
    }

    fn count(&self) -> f32 {
        self.count
    }

    fn min(&self) -> f32 {
        self.min
    }

    fn max(&self) -> f32 {
        self.max
    }
}

#[derive(Clone, Debug)]
pub struct TDigest {
    centroids: Vec<Centroid>,
    max_size: usize,
    /// use float here to avoid lots of usize <-> float conversions
    count: f32,
    sum: f32,
    // undefined for empty digest...
    // TODO(philiphayes): default 0.0? f32::NAN? None?
    min: f32,
    max: f32,
}

impl TDigest {
    pub fn new_with_size(max_size: usize) -> Self {
        Self {
            centroids: Vec::new(),
            max_size,
            count: 0.0,
            sum: 0.0,
            min: f32::NAN,
            max: f32::NAN,
        }
    }

    #[inline]
    pub fn mean(&self) -> f32 {
        if self.count > 0.0 {
            self.sum / self.count
        } else {
            // TODO(philiphayes): default 0.0? f32::NAN? None?
            0.0
        }
    }

    pub fn merge_unsorted(&mut self, values: &mut [f32]) {
        sort_unstable_f32(values);
        self.merge_sorted(values);
    }

    pub fn merge_sorted(&mut self, values: &[f32]) {
        debug_assert!(
            is_sorted_f32(values.iter().copied()),
            "values are not sorted: {:?}",
            values
        );

        if values.is_empty() {
            return;
        }

        let values_min = *values.first().unwrap();
        let values_max = *values.last().unwrap();
        let (total_min, total_max) = if self.count > 0.0 {
            let new_min = min_f32(self.min, values_min);
            let new_max = max_f32(self.max, values_max);
            (new_min, new_max)
        } else {
            (values_min, values_max)
        };

        let total_count = self.count + values.len() as f32;
        let mut total_sum = self.sum;

        let mut k_limit: f32 = 1.0;
        let mut q_limit_times_count = k_inv(k_limit, self.max_size as f32) * total_count;
        k_limit += 1.0;

        let mut compressed: Vec<Centroid> = Vec::with_capacity(self.max_size);

        let mut iter_centroids = self.centroids.iter().peekable();
        let mut iter_values = values.iter().peekable();

        let mut curr = if let Some(c) = iter_centroids.peek() {
            let curr = **iter_values.peek().unwrap();
            if c.mean < curr {
                *iter_centroids.next().unwrap()
            } else {
                Centroid::unit(*iter_values.next().unwrap())
            }
        } else {
            Centroid::unit(*iter_values.next().unwrap())
        };

        let mut prefix_count = curr.count;
        let mut sums_to_merge: f32 = 0.0;
        let mut counts_to_merge: f32 = 0.0;

        while iter_centroids.peek().is_some() || iter_values.peek().is_some() {
            let next = if let Some(c) = iter_centroids.peek() {
                if iter_values.peek().is_none() || c.mean < **iter_values.peek().unwrap() {
                    *iter_centroids.next().unwrap()
                } else {
                    Centroid::unit(*iter_values.next().unwrap())
                }
            } else {
                Centroid::unit(*iter_values.next().unwrap())
            };

            let next_sum = next.mean * next.count;
            prefix_count += next.count;

            if prefix_count <= q_limit_times_count {
                sums_to_merge += next_sum;
                counts_to_merge += next.count;
            } else {
                total_sum += curr.add(sums_to_merge, counts_to_merge);
                sums_to_merge = 0.0;
                counts_to_merge = 0.0;

                compressed.push(curr);
                q_limit_times_count = k_inv(k_limit, self.max_size as f32) * total_count;
                k_limit += 1.0;
                curr = next;
            }
        }

        compressed.push(curr);
        compressed.shrink_to_fit();
        // TODO(philiphayes): this should always be true...
        debug_assert!(is_sorted_by(compressed.iter(), |c1, c2| Some(
            total_cmp_f32(&c1.mean, &c2.mean)
        )));
        // TODO(philiphayes): which would imply we can remove this
        // compressed.sort_unstable_by(|c1, c2| total_cmp_f32(&c1.mean, &c2.mean));

        debug_assert!(compressed.len() <= self.max_size);

        self.centroids = compressed;
        self.count = total_count;
        self.sum = total_sum;
        self.min = total_min;
        self.max = total_max;
    }

    pub fn merge_v2(&mut self, digest: impl Digest) {
        if digest.is_empty() {
            return;
        }

        let (total_min, total_max) = if self.count > 0.0 {
            (
                min_f32(self.min, digest.min()),
                max_f32(self.max, digest.max()),
            )
        } else {
            (digest.min(), digest.max())
        };

        let total_count = self.count + digest.count();
        let mut total_sum = self.sum;

        let mut k_limit: f32 = 1.0;
        let mut q_limit_times_count = k_inv(k_limit, self.max_size as f32) * total_count;
        k_limit += 1.0;

        let centroids_1 = mem::replace(&mut self.centroids, Vec::with_capacity(self.max_size));
        let mut merged = itertools::merge(centroids_1, digest.centroids());

        let mut curr = merged.next().unwrap();
        let mut prefix_count: f32 = curr.count;
        let mut sums_to_merge: f32 = 0.0;
        let mut counts_to_merge: f32 = 0.0;

        for next in merged {
            let next_sum = next.mean * next.count;
            prefix_count += next.count;

            if prefix_count <= q_limit_times_count {
                sums_to_merge += next_sum;
                counts_to_merge += next.count;
            } else {
                total_sum += curr.add(sums_to_merge, counts_to_merge);
                sums_to_merge = 0.0;
                counts_to_merge = 0.0;

                self.centroids.push(curr);
                q_limit_times_count = k_inv(k_limit, self.max_size as f32) * total_count;
                k_limit += 1.0;
                curr = next;
            }
        }

        self.centroids.push(curr);
        self.centroids.shrink_to_fit();

        self.count = total_count;
        self.sum = total_sum;
        self.min = total_min;
        self.max = total_max;

        debug_assert!(self.centroids.len() <= self.max_size);
        debug_assert!(is_sorted_by(self.centroids.iter(), |c1, c2| Some(
            total_cmp_f32(&c1.mean, &c2.mean)
        )));
    }

    pub fn merge(&mut self, _digests: TDigest) {}

    #[inline]
    pub fn inv_cdf(&self, q: f32) -> f32 {
        self.quantile(q)
    }

    pub fn quantile(&self, q: f32) -> f32 {
        debug_assert!((0.0..=1.0).contains(&q));

        if self.centroids.is_empty() {
            // undefined for empty digest...
            // TODO(philiphayes): default 0.0? f32::NAN? None?
            return 0.0;
        }

        let rank = q * self.count;
        let mut pos: usize;
        let mut t: f32;

        if q > 0.5 {
            if q >= 1.0 {
                return self.max;
            }

            pos = 0;
            t = self.count;

            for (idx, centroid) in self.centroids.iter().enumerate().rev() {
                t -= centroid.count;

                if rank >= t {
                    pos = idx;
                    break;
                }
            }
        } else {
            if q <= 0.0 {
                return self.min;
            }

            pos = self.centroids.len() - 1;
            t = 0.0;

            for (idx, centroid) in self.centroids.iter().enumerate() {
                if rank < t + centroid.count {
                    pos = idx;
                    break;
                }

                t += centroid.count;
            }
        }

        let mid_mean = self.centroids[pos].mean;
        let mid_count = self.centroids[pos].count;

        let (min, max, delta) = if self.centroids.len() > 1 {
            if pos == 0 {
                let max = self.centroids[pos + 1].mean;
                let delta = max - mid_mean;
                (self.min, max, delta)
            } else if pos == self.centroids.len() - 1 {
                let min = self.centroids[pos - 1].mean;
                let delta = mid_mean - min;
                (min, self.max, delta)
            } else {
                let min = self.centroids[pos - 1].mean;
                let max = self.centroids[pos + 1].mean;
                let delta = 0.5 * (max - min);
                (min, max, delta)
            }
        } else {
            return mid_mean;
        };

        let quantile = mid_mean + (((rank - t) / mid_count) - 0.5) * delta;

        // println!(
        //     "quantile: n: {:?}, rank: {:?}, t: {:?}, pos: {}, delta: {}, out: {}",
        //     self.count, rank, t, pos, delta, quantile
        // );

        clamp_f32(quantile, min, max)
    }

    pub fn quantile_v2(&self, q: f32) -> f32 {
        debug_assert!((0.0..=1.0).contains(&q));

        let n = self.centroids.len();

        if n == 0 {
            return 0.0;
        } else if n == 1 {
            return self.centroids[0].mean;
        }
        let rank = q * self.count;

        // println!(
        //     "quantile_v2: q: {:?}, n: {}, count: {:?}, rank: {:?}",
        //     q, n, self.count, rank
        // );

        // we're looking for the first element (minimum)
        if rank < 1.0 {
            return self.min;
        }

        // we're somewhere between the minimum and the first centroid
        if !self.centroids[0].is_unit() && rank < 0.5 * self.centroids[0].count {
            // lerp b/w min when rank <- 1.0, c_0.mean when rank -> c_0.count / 2
            let t = (rank - 1.0) / ((0.5 * self.centroids[0].count) - 1.0);
            return lerp_f32(self.min, self.centroids[0].mean, t);
        }

        // we're looking for the last element (maximum)
        if rank > self.count - 1.0 {
            return self.max;
        }

        // we're somewhere between the last centroid and the maximum
        if !self.centroids[n - 1].is_unit()
            && self.count - rank <= 0.5 * self.centroids[n - 1].count
        {
            // lerp b/w the last centroid and the max. we lerp down to avoid some
            // extra ops. (lerp(x0, x1, t) == lerp(x1, x0, 1 - t))
            let t = (self.count - rank - 1.0) / ((0.5 * self.centroids[n - 1].count) - 1.0);
            return lerp_f32(self.max, self.centroids[n - 1].mean, t);
        }

        // count_so_far is center of centroid_i
        let mut count_so_far = 0.5 * self.centroids[0].count;

        for i in 0..n - 1 {
            let delta_count = 0.5 * (self.centroids[i].count + self.centroids[i + 1].count);

            // println!(
            //     "count_so_far: {:?}, delta_count: {:?}, bound: {:?}",
            //     count_so_far,
            //     delta_count,
            //     count_so_far + delta_count
            // );

            if count_so_far + delta_count > rank {
                // rank is somewhere between centroid_i and centroid_i+1

                let left_unit = if self.centroids[i].is_unit() {
                    // centroid_i is a unit centroid (only one sample)
                    if rank - count_so_far < 0.5 {
                        // our rank is inside the unit centroid
                        return self.centroids[i].mean;
                    } else {
                        0.5
                    }
                } else {
                    0.0
                };

                let right_unit = if self.centroids[i + 1].is_unit() {
                    // centroid_i+1 is a unit centroid (only one sample)
                    if count_so_far + delta_count - rank <= 0.5 {
                        // our rank is inside the unit centroid
                        return self.centroids[i + 1].mean;
                    } else {
                        0.5
                    }
                } else {
                    0.0
                };

                // dbg!(left_unit);
                // dbg!(right_unit);

                // distance from center of centroid_i to rank
                let d1 = rank - count_so_far - left_unit;
                // distance from rank to center of centroid_i+1
                let d2 = count_so_far + delta_count - rank - right_unit;
                let t = d1 / (d1 + d2);
                let u1 = self.centroids[i].mean;
                let u2 = self.centroids[i + 1].mean;
                return lerp_clamp_f32(u1, u2, t);
            }

            count_so_far += delta_count;
        }

        // println!("count_so_far: {:?}", count_so_far);

        // I don't believe this is always true unless `rank = q * (n - 1)`
        // debug_assert!(!self.centroids[n - 1].is_unit());
        debug_assert!(rank <= self.count);
        // debug_assert!(rank >= self.count - 0.5 * self.centroids[n - 1].count);
        debug_assert!(rank >= count_so_far);

        // distance from center of last centroid to rank. we could also use
        // `rank - count_so_far` but this has less accumulated fp error.
        // let d1 = rank - (self.count - 0.5 * self.centroids[n - 1].count);
        // let t = d1 / (0.5 * self.centroids[n - 1].count);
        let d1 = rank - count_so_far;
        let t = d1 / (self.count - count_so_far);

        lerp_clamp_f32(self.centroids[n - 1].mean, self.max, t)
    }

    pub fn quantile_v3(&self, q: f32) -> f32 {
        debug_assert!((0.0..=1.0).contains(&q));

        let n = self.centroids.len();

        if n == 0 {
            // TODO(philiphayes): technically undefined. return NAN? None?
            return 0.0;
        }
        if n == 1 {
            return self.centroids[0].mean;
        }
        if q <= 0.0 {
            return self.min;
        }
        if q >= 1.0 {
            return self.max;
        }

        let max_rank = self.count;
        let rank = q * max_rank;

        // println!(
        //     "quantile_v3: q: {:?}, n: {}, max_rank: {:?}, rank: {:?}",
        //     q, n, max_rank, rank
        // );

        debug_assert!(rank <= self.count);

        if rank < 0.5 + 0.5 * self.centroids[0].count {
            let t = rank / (0.5 + 0.5 * self.centroids[0].count);
            return lerp_clamp_f32(self.min, self.centroids[0].mean, t);
        }

        if rank >= max_rank - 0.5 - 0.5 * self.centroids[n - 1].count {
            let t = 1.0 - ((max_rank - rank) / (0.5 + 0.5 * self.centroids[n - 1].count));
            return lerp_clamp_f32(self.centroids[n - 1].mean, self.max, t);
        }

        let mut count_so_far = 0.5 + 0.5 * self.centroids[0].count;
        for i in 0..n - 1 {
            let delta_count = 0.5 * (self.centroids[i].count + self.centroids[i + 1].count);

            // println!(
            //     "count_so_far: {:?}, delta_count: {:?}",
            //     count_so_far, delta_count
            // );

            if rank < count_so_far + delta_count {
                let t = (rank - count_so_far) / delta_count;
                return lerp_clamp_f32(self.centroids[i].mean, self.centroids[i + 1].mean, t);
            }

            count_so_far += delta_count;
        }

        debug_assert!(rank >= count_so_far);
        // debug_assert!(count_so_far <= max_rank);
        debug_assert!(count_so_far <= max_rank - 0.5);

        // can fall through due to accumulated float errors in count_so_far.
        let t = (rank - count_so_far) / (max_rank - count_so_far);
        lerp_clamp_f32(self.centroids[n - 1].mean, self.max, t)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::{assert_abs_diff_eq, assert_relative_eq, assert_ulps_eq};
    use itertools::Itertools;
    use proptest::{collection::vec, prelude::*, sample::subsequence};
    use rand::{distributions::Distribution, rngs::SmallRng, SeedableRng};
    use statrs::distribution::{Exponential, Normal};
    use std::{cmp::min, fmt::Debug, ops::Range};

    fn arb_small_rng() -> impl Strategy<Value = SmallRng> {
        any::<u64>().prop_map(SmallRng::seed_from_u64).no_shrink()
    }

    fn small_rng() -> SmallRng {
        SmallRng::seed_from_u64(0xdeadb33f)
    }

    fn slices_total_eq_f32(left: &[f32], right: &[f32]) -> bool {
        if left.len() != right.len() {
            return false;
        }

        left.iter()
            .zip(right.iter())
            .all(|(a, b)| matches!(total_cmp_f32(a, b), Ordering::Equal))
    }

    fn assert_slices_total_eq_f32(left: &[f32], right: &[f32]) {
        assert!(
            slices_total_eq_f32(left, right),
            "(left == right)\n  left: {:?}\n right: {:?}",
            left,
            right
        );
    }

    fn assert_slices_total_ne_f32(left: &[f32], right: &[f32]) {
        assert!(
            !slices_total_eq_f32(left, right),
            "(left != right)\n  left: {:?}\n right: {:?}",
            left,
            right
        );
    }

    fn linspace_incl(start: f32, stop: f32, num: usize) -> impl Iterator<Item = f32> {
        assert!(num >= 2);

        let div = (num - 1) as f32;
        let delta = stop - start;
        let step = delta / div;

        (0..num).map(move |x| start + (x as f32) * step)
    }

    // fn linspace_excl(start: f32, stop: f32, num: usize) -> impl Iterator<Item = f32> {
    //     assert!(num >= 1);
    //
    //     let div = num as f32;
    //     let delta = stop - start;
    //     let step = delta / div;
    //
    //     (0..num).map(move |x| start + (x as f32) * step)
    // }

    // effectively equivalent to `TDigest::quantile`
    #[allow(dead_code)]
    fn empirical_quantile(xs: &[f32], q: f32) -> f32 {
        assert!((0.0..=1.0).contains(&q));
        assert!(!xs.is_empty());

        let n = xs.len();
        let rank = q * n as f32;
        let idx_mid = min(rank.floor() as usize, n - 1);

        if n == 1 {
            return xs[0];
        }

        let (x_min, x_max, delta) = if idx_mid == 0 {
            (xs[0], xs[1], xs[1] - xs[0])
        } else if idx_mid == n - 1 {
            (xs[n - 2], xs[n - 1], xs[n - 1] - xs[n - 2])
        } else {
            (
                xs[idx_mid - 1],
                xs[idx_mid + 1],
                0.5 * (xs[idx_mid + 1] - xs[idx_mid - 1]),
            )
        };

        let x_out = xs[idx_mid] + delta * (rank - rank.floor() - 0.5);
        clamp_f32(x_out, x_min, x_max)
    }

    // equivalent to:
    // `scipy.stats.mstats.mquantiles(xs, prob = p, alphap = 0.0, betap = 1.0)`
    #[allow(dead_code)]
    fn empirical_quantile_a0b1(sorted_values: &[f32], q: f32) -> f32 {
        assert!((0.0..=1.0).contains(&q));
        assert!(!sorted_values.is_empty());

        if q <= 0.0 {
            return *sorted_values.first().unwrap();
        } else if q >= 1.0 {
            return *sorted_values.last().unwrap();
        }

        let n = sorted_values.len() as f32;
        let idx = q * (n - 1.0);
        let idx_lo = idx.floor() as usize;
        let idx_hi = idx.ceil() as usize;

        // println!("empirical_quantile: idx: {:?}", idx);

        if idx_lo == idx_hi {
            // we happen to be on an integer boundary
            sorted_values[idx_lo]
        } else {
            // linearly interpolate between adjacent values
            lerp_f32(sorted_values[idx_lo], sorted_values[idx_hi], idx.fract())
        }
    }

    // effectively equivalent to `TDigest::quantile_v3`
    fn empirical_quantile_v3(xs: &[f32], q: f32) -> f32 {
        assert!((0.0..=1.0).contains(&q));
        assert!(!xs.is_empty());

        let n = xs.len();
        let rank = q * n as f32;

        if rank < 1.0 {
            return xs[0];
        } else if rank >= (n as f32) - 1.0 {
            return xs[n - 1];
        }

        let idx = (rank - 1.0) as usize;
        lerp_clamp_f32(xs[idx], xs[idx + 1], rank.fract())
    }

    fn sample_distr(
        rng: &mut SmallRng,
        distr: impl Distribution<f64>,
        num_samples: usize,
    ) -> Vec<f32> {
        distr
            .sample_iter(rng)
            .take(num_samples)
            .map(|x| x as f32)
            .collect::<Vec<_>>()
    }

    fn arb_partitions(
        len: usize,
    ) -> impl Strategy<Value = impl Iterator<Item = Range<usize>> + Debug> {
        let idxs = (1..len).collect::<Vec<_>>();
        subsequence(idxs, 0..len).prop_map(move |mut idxs| {
            idxs.push(len);
            idxs.into_iter().scan(0_usize, |prev, curr| {
                let range = *prev..curr;
                *prev = curr;
                Some(range)
            })
        })
    }

    fn arb_partitioned_samples(
        distr: impl Distribution<f64> + Copy,
        max_samples: usize,
    ) -> impl Strategy<Value = (Vec<f32>, impl Iterator<Item = Range<usize>> + Debug)> {
        ((1..max_samples), arb_small_rng()).prop_flat_map(move |(num_samples, mut rng)| {
            let samples = sample_distr(&mut rng, distr, num_samples);
            let partitions = arb_partitions(num_samples);
            (Just(samples), partitions)
        })
    }

    #[test]
    fn test_t_digest_quantile_lossless() {
        let mut digest = TDigest::new_with_size(5);
        digest.centroids = vec![
            Centroid::unit(1.0),
            Centroid::unit(2.0),
            Centroid::unit(3.0),
        ];
        digest.count = 3.0;
        digest.sum = 1.0 + 2.0 + 3.0;
        digest.min = 1.0;
        digest.max = 3.0;

        let values = digest.centroids.iter().map(|c| c.mean).collect::<Vec<_>>();

        for q in linspace_incl(0.0, 1.0, 20) {
            assert_abs_diff_eq!(
                empirical_quantile_v3(&values, q),
                digest.quantile_v3(q),
                epsilon = 1e-5,
            );
        }
    }

    #[test]
    fn test_t_digest_basic_range() {
        let mut digest = TDigest::new_with_size(100);
        let mut digest2 = TDigest::new_with_size(100);

        let n_u16 = 10_000_u16;
        let n = n_u16 as f32;
        let values = (1..=n_u16).map(f32::from).collect::<Vec<_>>();

        digest.merge_sorted(&values);
        digest2.merge_v2(values.as_slice());

        assert_relative_eq!(0.99 * n, digest.quantile_v3(0.99), max_relative = 1e-3);
        assert_relative_eq!(0.99 * n, digest2.quantile_v3(0.99), max_relative = 1e-3);

        assert_ulps_eq!(1.0, digest.min);
        assert_ulps_eq!(1.0, digest2.min);

        assert_ulps_eq!(n, digest.max);
        assert_ulps_eq!(n, digest2.max);

        assert_ulps_eq!(n, digest.count);
        assert_ulps_eq!(n, digest2.count);

        // sum_{i=1}^n i = n(n+1)/2
        assert_relative_eq!(0.5 * n * (n + 1.0), digest.sum, max_relative = 1e-3);
        assert_relative_eq!(0.5 * n * (n + 1.0), digest2.sum, max_relative = 1e-3);

        for q in linspace_incl(0.0, 1.0, (n_u16 + 1) as usize) {
            assert_abs_diff_eq!(
                empirical_quantile_v3(&values, q),
                digest.quantile_v3(q),
                epsilon = 1.0,
            );
            assert_ulps_eq!(digest.quantile_v3(q), digest2.quantile_v3(q));
        }
    }

    #[test]
    fn test_t_digest_exp_cdf_approx() {
        let mut rng = small_rng();
        let exp_distr = Exponential::new(1.0).unwrap();

        let mut samples = sample_distr(&mut rng, exp_distr, 10_000);
        sort_unstable_f32(&mut samples);

        let mut digest = TDigest::new_with_size(100);
        let mut digest2 = TDigest::new_with_size(100);
        digest.merge_sorted(&samples);
        digest2.merge_v2(samples.as_slice());

        for q in linspace_incl(0.0, 1.0, 10000 + 1) {
            assert_abs_diff_eq!(
                empirical_quantile_v3(&samples, q),
                digest.quantile_v3(q),
                epsilon = 0.5,
            );
            assert_ulps_eq!(digest.quantile_v3(q), digest2.quantile_v3(q));
        }
    }

    #[test]
    fn test_prop_regression() {
        #[rustfmt::skip]
        let mut samples: Vec<f32> = vec![
            0.19474669, 0.51428425, 1.4059021, 0.9500685, 1.0093654, 2.2817805,
            0.039707553, 0.612078, 0.99576074, 0.12096588, 1.0654054, 0.84764045,
            1.7110986, 0.0051696994, 0.7866332, 0.59930784, 0.10637737, 1.3672066,
            0.5911403, 1.4906442, 0.22215544, 1.0755656, 0.3992901, 0.39417312,
            0.24705158, 0.3845353, 1.1957443, 0.34679464, 0.044407744, 1.2291212,
            0.3868451, 0.58477145, 1.5412507, 0.13874866, 2.4159024, 0.6062898,
            0.45989984, 0.20425297, 1.5605284, 1.2888067, 0.034640927, 2.4805949,
            0.9771346, 1.9280208, 0.18084313, 0.42018482, 0.3574236, 0.41065904,
            0.5239536, 1.4164546, 0.52283335, 0.5395101, 0.93963134, 0.1854061,
            2.7068782, 0.7427874, 0.5134245, 0.39419377, 2.6069608, 0.063637555,
            0.72978956, 1.0464643, 0.35273927, 0.49700767, 0.016751215, 0.3237345,
            0.10114882, 0.65465933, 0.45219454, 1.9272926, 0.14119564, 1.2539241,
            0.07170107, 0.06633472, 0.38315305, 0.38155735, 0.5192001, 1.2789347,
            0.15902951, 0.8302144, 0.043188743, 0.32083005, 0.26170507, 0.2654353,
            0.22921032, 7.2142696, 0.8015038, 0.24489278, 1.9354001, 0.16499609,
            0.23848225, 0.39285257, 0.5192319, 2.0908654, 2.402804, 0.6102857,
            1.8426602, 2.069859, 0.0002470554, 0.6629204, 0.29940364, 0.45412397,
            0.3401415, 0.6725147, 0.48499689, 0.24618141, 1.0014629, 0.76581347,
            0.6985204, 0.36325228, 0.01570845, 0.6050948, 2.0912147, 0.37514824,
            0.8307331, 0.30853304, 1.0640845, 1.3774515, 0.47575802, 0.34704444,
            0.15182784, 0.20800605, 0.12036719, 0.3738441, 0.1645731, 2.306107,
            1.1633161, 2.6265442, 1.4233699, 1.7409527, 2.210432, 0.37948158,
            1.5684712, 0.5906306, 1.3305123, 0.4077533, 0.12074636, 0.3278894, 4.973523
        ];
        sort_unstable_f32(&mut samples);

        let mut digest = TDigest::new_with_size(100);
        let mut digest2 = TDigest::new_with_size(100);
        digest.merge_sorted(&samples);
        digest2.merge_v2(samples.as_slice());

        for q in linspace_incl(0.0, 1.0, 100 + 1) {
            assert_abs_diff_eq!(
                empirical_quantile_v3(&samples, q),
                digest.quantile_v3(q),
                epsilon = 0.1,
            );
            assert_ulps_eq!(digest.quantile_v3(q), digest2.quantile_v3(q));
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn test_t_digest_exp_error_bound(
            (mut samples, partitions) in arb_partitioned_samples(Exponential::new(1.0).unwrap(), 1000)
        ) {
            let mut digest = TDigest::new_with_size(32);
            let mut digest2 = TDigest::new_with_size(32);

            for range in partitions {
                let partition = &mut samples[range];
                digest.merge_unsorted(partition);
                digest2.merge_v2(&*partition);

                assert_ulps_eq!(digest.min, digest2.min);
                assert_ulps_eq!(digest.max, digest2.max);
                assert_ulps_eq!(digest.sum, digest2.sum);
                assert_ulps_eq!(digest.count, digest2.count);
            }

            sort_unstable_f32(&mut samples);

            for q in linspace_incl(0.0, 1.0, 100 + 1) {
                assert_abs_diff_eq!(
                    digest.quantile_v3(q),
                    digest2.quantile_v3(q),
                    epsilon = 1e-3,
                );
                assert_abs_diff_eq!(
                    empirical_quantile_v3(&samples, q),
                    digest.quantile_v3(q),
                    epsilon = 1.5,
                );
            }
        }

        #[test]
        fn test_t_digest_gaussian_error_bound(
            (mut samples, partitions) in arb_partitioned_samples(Normal::new(0.0, 1.0).unwrap(), 1000)
        ) {
            let mut digest = TDigest::new_with_size(32);
            let mut digest2 = TDigest::new_with_size(32);

            for range in partitions {
                let partition = &mut samples[range];
                digest.merge_unsorted(partition);
                digest2.merge_v2(&*partition);

                assert_ulps_eq!(digest.min, digest2.min);
                assert_ulps_eq!(digest.max, digest2.max);
                assert_ulps_eq!(digest.sum, digest2.sum);
                assert_ulps_eq!(digest.count, digest2.count);
            }

            sort_unstable_f32(&mut samples);

            for q in linspace_incl(0.0, 1.0, 100 + 1) {
                assert_abs_diff_eq!(
                    digest.quantile_v3(q),
                    digest2.quantile_v3(q),
                    epsilon = 1e-3,
                );
                assert_abs_diff_eq!(
                    empirical_quantile_v3(&samples, q),
                    digest.quantile_v3(q),
                    epsilon = 0.5,
                );
            }
        }

        #[test]
        fn test_t_digest_merge_digest(
            (mut samples, partitions) in arb_partitioned_samples(Exponential::new(1.0).unwrap(), 1000)
        ) {
            let mut digests = Vec::new();

            for range in partitions {
                let partition = &mut samples[range];
                let mut digest = TDigest::new_with_size(32);
                sort_unstable_f32(partition);
                digest.merge_v2(&*partition);
                digests.push(digest);
            }

            let merged_digest = digests.into_iter()
                .tree_fold1(|mut d1, d2| {
                    d1.merge_v2(d2);
                    d1
                })
                .unwrap();
            sort_unstable_f32(&mut samples);

            let mut digest = TDigest::new_with_size(32);
            digest.merge_v2(samples.as_slice());

            for q in linspace_incl(0.0, 1.0, 100 + 1) {
                assert_abs_diff_eq!(
                    empirical_quantile_v3(&samples, q),
                    merged_digest.quantile_v3(q),
                    epsilon = 2.0,
                );

                assert_abs_diff_eq!(
                    digest.quantile_v3(q),
                    merged_digest.quantile_v3(q),
                    epsilon = 2.0,
                );
            }
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(1000))]

        #[test]
        fn test_is_sorted_equiv(array in vec(any::<f32>(), 0..100)) {
            let mut sorted_array = array.clone();
            sort_unstable_f32(&mut sorted_array);
            assert!(is_sorted_f32(sorted_array.iter().copied()));

            // array.is_sorted <==> array == sorted_array
            if is_sorted_f32(array.iter().copied()) {
                assert_slices_total_eq_f32(&array, &sorted_array);
            } else {
                assert_slices_total_ne_f32(&array, &sorted_array);
            }
        }
    }
}
