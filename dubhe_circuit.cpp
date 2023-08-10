#include "dubhe.h"
#include "circuit.h"

// #define TIMING
#ifdef TIMING
#include "tools/bench_timing.h"
#endif

#include "aes.h"
#include "field.h"
#include "tape.h"
#include "tree.h"
#include <algorithm>
#include <cassert>
#include <cstring>

extern "C" {
#include "kdf_shake.h"
#include "randomness.h"
}

// void print(__m256i x)
// {
//     uint8_t tmp[32];
//     _mm256_storeu_si256((__m256i *)tmp, x);
//     for (int i = 0; i < 32; i++) {
//         printf("%02x ", tmp[i]);
//     }
//     printf("\n");
//     return;
// }
// 
// void print(const field::GF2E &x)
// {
//     printf("0x%04x", (uint16_t)x.data);
//     return;
// }

inline uint16_t _mm_hxor_epu16(__m128i x)
{
    // https://stackoverflow.com/questions/6996764/fastest-way-to-do-horizontal-sse-vector-sum-or-other-reduction
    __m128i hi64 = _mm_unpackhi_epi64(x, x);
    __m128i xor6 = _mm_xor_si128(hi64, x);
    __m128i hi32 = _mm_shufflelo_epi16(xor6, _MM_SHUFFLE(1, 0, 3, 2));
    __m128i xor3 = _mm_xor_si128(xor6, hi32);
    uint32_t xor32 = _mm_cvtsi128_si32(xor3);
    uint16_t xor16 = ((uint16_t) xor32) ^ (xor32 >> 16);
    return ((uint8_t) xor16) ^ (xor16 >> 8);
}

inline uint16_t  _mm256_hxor_epu16(__m256i x)
{
    return _mm_hxor_epu16(_mm256_castsi256_si128(x)) |
        (_mm_hxor_epu16(_mm256_extracti128_si256(x, 1)) << 8);
}

namespace {
inline void hash_update_GF2E(hash_context *ctx,
                             const dubhe_instance_t &instance,
                             const field::GF2E &element) {
  // 8 bytes is enough for supported field sizes
  std::array<uint8_t, 8> buffer;
  element.to_bytes(buffer.data());
  hash_update(ctx, buffer.data(), instance.lambda);
}

// void print_hash(const std::vector<uint8_t> hash)
// {
//     for (auto b: hash)
//         printf("%02x", b);
//     printf("\n");
// }

std::pair<dubhe_salt_t, std::vector<std::vector<uint8_t>>>
generate_salt_and_seeds(const dubhe_instance_t &instance) {
    // salt, seed_1, ..., seed_r = H(instance||sk||pk||m)
    hash_context ctx;
    hash_init(&ctx, instance.digest_size);
    hash_update_uint16_le(&ctx, (uint16_t)instance.params);
    hash_final(&ctx);

    dubhe_salt_t salt;
    hash_squeeze(&ctx, salt.data(), salt.size());
    std::vector<std::vector<uint8_t>> seeds;
    for (size_t repetition = 0; repetition < instance.num_rounds; repetition++) {
        std::vector<uint8_t> s(instance.seed_size);
        hash_squeeze(&ctx, s.data(), s.size());
        seeds.push_back(s);
    }
    return std::make_pair(salt, seeds);
}

void commit_to_party_seed(const dubhe_instance_t &instance,
                          const gsl::span<uint8_t> &seed,
                          const dubhe_salt_t &salt, size_t rep_idx,
                          size_t party_idx, gsl::span<uint8_t> commitment) {
  hash_context ctx;
  hash_init(&ctx, instance.digest_size);
  hash_update(&ctx, salt.data(), salt.size());
  hash_update_uint16_le(&ctx, (uint16_t)rep_idx);
  hash_update_uint16_le(&ctx, (uint16_t)party_idx);
  hash_update(&ctx, seed.data(), seed.size());
  hash_final(&ctx);

  hash_squeeze(&ctx, commitment.data(), commitment.size());
}

void commit_to_4_party_seeds(
    const dubhe_instance_t &instance, const gsl::span<uint8_t> &seed0,
    const gsl::span<uint8_t> &seed1, const gsl::span<uint8_t> &seed2,
    const gsl::span<uint8_t> &seed3, const dubhe_salt_t &salt, size_t rep_idx,
    size_t party_idx, gsl::span<uint8_t> com0, gsl::span<uint8_t> com1,
    gsl::span<uint8_t> com2, gsl::span<uint8_t> com3) {
  hash_context_x4 ctx;
  hash_init_x4(&ctx, instance.digest_size);
  hash_update_x4_1(&ctx, salt.data(), salt.size());
  hash_update_x4_uint16_le(&ctx, (uint16_t)rep_idx);
  const uint16_t party_idxs[4] = {
      (uint16_t)party_idx, (uint16_t)(party_idx + 1), (uint16_t)(party_idx + 2),
      (uint16_t)(party_idx + 3)};
  hash_update_x4_uint16s_le(&ctx, party_idxs);
  hash_update_x4_4(&ctx, seed0.data(), seed1.data(), seed2.data(), seed3.data(),
                   instance.seed_size);
  hash_final_x4(&ctx);

  hash_squeeze_x4_4(&ctx, com0.data(), com1.data(), com2.data(), com3.data(),
                    instance.digest_size);
}

std::vector<uint8_t>
phase_1_commitment(const dubhe_instance_t &instance,
                   const dubhe_salt_t &salt,
                   const RepByteContainer &commitments,
                   const std::vector<std::vector<uint8_t>> &key_deltas) {

  hash_context ctx;
  hash_init_prefix(&ctx, instance.digest_size, HASH_PREFIX_1);
  hash_update(&ctx, salt.data(), salt.size());

  for (size_t repetition = 0; repetition < instance.num_rounds; repetition++) {
    for (size_t party = 0; party < instance.num_MPC_parties; party++) {
      auto commitment = commitments.get(repetition, party);
      hash_update(&ctx, commitment.data(), commitment.size());
    }
    hash_update(&ctx, key_deltas[repetition].data(),
                key_deltas[repetition].size());
  }
  hash_final(&ctx);

  std::vector<uint8_t> commitment(instance.digest_size);
  hash_squeeze(&ctx, commitment.data(), commitment.size());
  return commitment;
}

std::vector<uint8_t>
group_phase_1_commitment(const dubhe_instance_t &instance,
                   const dubhe_salt_t &salt,
                   const uint8_t *message, size_t message_len,
                   const RepByteContainer &commitments,
                   const std::vector<std::vector<uint8_t>> &key_deltas,
                   const std::vector<std::vector<uint8_t>> &t_deltas,
                   const std::vector<std::vector<uint8_t>> &pt_deltas) {

  hash_context ctx;
  hash_init_prefix(&ctx, instance.digest_size, HASH_PREFIX_1);
  hash_update(&ctx, salt.data(), salt.size());
  hash_update(&ctx, message, message_len);

  for (size_t repetition = 0; repetition < instance.num_rounds; repetition++) {
    for (size_t party = 0; party < instance.num_MPC_parties; party++) {
      auto commitment = commitments.get(repetition, party);
      hash_update(&ctx, commitment.data(), commitment.size());
      // auto output_broadcast = output_broadcasts.get(repetition, party);
      // hash_update(&ctx, output_broadcast.data(), output_broadcast.size());
    }
    hash_update(&ctx, key_deltas[repetition].data(), key_deltas[repetition].size());
    hash_update(&ctx, t_deltas[repetition].data(), t_deltas[repetition].size());
    hash_update(&ctx, pt_deltas[repetition].data(), pt_deltas[repetition].size());
  }
  hash_final(&ctx);

  std::vector<uint8_t> commitment(instance.digest_size);
  hash_squeeze(&ctx, commitment.data(), commitment.size());
  return commitment;
}

// std::vector<std::vector<field::GF2E>>
// phase_1_expand(const dubhe_instance_t &instance,
//                const std::vector<uint8_t> &h_1) {
//   hash_context ctx;
//   hash_init(&ctx, instance.digest_size);
//   hash_update(&ctx, h_1.data(), h_1.size());
//   hash_final(&ctx);
// 
//   std::vector<std::vector<field::GF2E>> r_ejs;
//   std::vector<uint8_t> lambda_sized_buffer(instance.lambda);
//   r_ejs.reserve(instance.num_rounds);
//   for (size_t e = 0; e < instance.num_rounds; e++) {
//     std::vector<field::GF2E> r_js;
//     r_js.resize(instance.m1);
//     for (size_t j = 0; j < instance.m1; j++) {
//       hash_squeeze(&ctx, lambda_sized_buffer.data(),
//                    lambda_sized_buffer.size());
//       r_js[j].from_bytes(lambda_sized_buffer.data());
//     }
//     r_ejs.push_back(r_js);
//   }
//   return r_ejs;
// }

std::vector<std::vector<field::GF2E>>
phase_sumcheck0_expand(const dubhe_instance_t &instance,
        const Circuit<GF2E> &C,
        const std::vector<uint8_t> &h_1) {
  hash_context ctx;
  hash_init(&ctx, instance.digest_size);
  hash_update(&ctx, h_1.data(), h_1.size());
  hash_final(&ctx);

  std::vector<std::vector<field::GF2E>> tau(C.layers[C.depth-1].bit_len);
  std::vector<uint8_t> lambda_sized_buffer(instance.lambda);

  for (auto &tau_i: tau) {
      tau_i.resize(instance.num_rounds);
      for (auto &tau_ie: tau_i) {
          hash_squeeze(&ctx, lambda_sized_buffer.data(), lambda_sized_buffer.size());
          tau_ie.from_bytes(lambda_sized_buffer.data());
      }
  }
  return tau;
}

std::vector<field::GF2E>
phase_sumcheck_expand(const dubhe_instance_t &instance,
        const std::vector<uint8_t> &h_1) {
  hash_context ctx;
  hash_init(&ctx, instance.digest_size);
  hash_update(&ctx, h_1.data(), h_1.size());
  hash_final(&ctx);

  std::vector<field::GF2E> r(instance.num_rounds);
  std::vector<uint8_t> lambda_sized_buffer(instance.lambda);

  for (auto &r_i: r) {
      hash_squeeze(&ctx, lambda_sized_buffer.data(), lambda_sized_buffer.size());
      r_i.from_bytes(lambda_sized_buffer.data());
  }
  return r;
}

std::vector<std::vector<field::GF2E>>
phase_sumcheck_final_expand(const dubhe_instance_t &instance,
        const std::vector<uint8_t> &h_1) {
    hash_context ctx;
    hash_init(&ctx, instance.digest_size);
    hash_update(&ctx, h_1.data(), h_1.size());
    hash_final(&ctx);

    std::vector<std::vector<field::GF2E>> scalar(instance.num_rounds);
    std::vector<uint8_t> lambda_sized_buffer(instance.lambda);

    for (auto &s_i: scalar) {
        s_i.resize(2);
        hash_squeeze(&ctx, lambda_sized_buffer.data(), lambda_sized_buffer.size());
        s_i[0].from_bytes(lambda_sized_buffer.data());
        hash_squeeze(&ctx, lambda_sized_buffer.data(), lambda_sized_buffer.size());
        s_i[1].from_bytes(lambda_sized_buffer.data());
    }
    return scalar;
}

std::vector<uint8_t>
phase_group1_expand(const dubhe_instance_t &instance,
        const std::vector<uint8_t> &h_1, const int size) {
  hash_context ctx;
  hash_init(&ctx, instance.digest_size);
  hash_update(&ctx, h_1.data(), h_1.size());
  hash_final(&ctx);

  std::vector<uint8_t> result(size * 2 * instance.aes_params.block_size * instance.aes_params.num_blocks);
  hash_squeeze(&ctx, result.data(), result.size());

  return result;
}

std::vector<uint8_t>
phase_sumcheck_commitment(const dubhe_instance_t &instance,
                   const dubhe_salt_t &salt, const std::vector<uint8_t> &h,
                   const std::vector<std::vector<field::GF2E>> &coef_deltas) {
    hash_context ctx;
    hash_init_prefix(&ctx, instance.digest_size, HASH_PREFIX_2);
    hash_update(&ctx, salt.data(), salt.size());
    hash_update(&ctx, h.data(), h.size());

    for (auto & cde : coef_deltas)
        for (auto & cdei : cde)
            hash_update_GF2E(&ctx, instance, cdei);

    hash_final(&ctx);

    std::vector<uint8_t> commitment(instance.digest_size);
    hash_squeeze(&ctx, commitment.data(), commitment.size());
    return commitment;
}

std::vector<uint8_t>
phase_sumcheck_final_commitment(const dubhe_instance_t &instance,
                   const dubhe_salt_t &salt, const std::vector<uint8_t> &h,
                   const std::vector<field::GF2E> &Vr_deltas,
                   const std::vector<field::GF2E> &Vs_deltas) {
    hash_context ctx;
    hash_init_prefix(&ctx, instance.digest_size, HASH_PREFIX_2);
    hash_update(&ctx, salt.data(), salt.size());
    hash_update(&ctx, h.data(), h.size());

    for (auto & cde : Vr_deltas)
        hash_update_GF2E(&ctx, instance, cde);
    for (auto & cde : Vs_deltas)
        hash_update_GF2E(&ctx, instance, cde);

    hash_final(&ctx);

    std::vector<uint8_t> commitment(instance.digest_size);
    hash_squeeze(&ctx, commitment.data(), commitment.size());
    return commitment;
}

std::vector<uint8_t>
phase_2_commitment(const dubhe_instance_t &instance,
                   const dubhe_salt_t &salt, const std::vector<uint8_t> &h_1,
                   const std::vector<std::vector<field::GF2E>> &P_deltas) {

  hash_context ctx;
  hash_init_prefix(&ctx, instance.digest_size, HASH_PREFIX_2);
  hash_update(&ctx, salt.data(), salt.size());
  hash_update(&ctx, h_1.data(), h_1.size());

  for (size_t repetition = 0; repetition < instance.num_rounds; repetition++) {
    for (size_t k = 0; k <= 2 * 1; k++) {
      hash_update_GF2E(&ctx, instance, P_deltas[repetition][k]);
    }
  }
  hash_final(&ctx);

  std::vector<uint8_t> commitment(instance.digest_size);
  hash_squeeze(&ctx, commitment.data(), commitment.size());
  return commitment;
}

std::vector<field::GF2E>
phase_2_expand(const dubhe_instance_t &instance,
               const std::vector<uint8_t> &h_2,
               const std::vector<field::GF2E> &forbidden_values) {
  hash_context ctx;
  hash_init(&ctx, instance.digest_size);
  hash_update(&ctx, h_2.data(), h_2.size());
  hash_final(&ctx);

  std::vector<uint8_t> lambda_sized_buffer(instance.lambda);
  std::vector<field::GF2E> R_es;
  for (size_t e = 0; e < instance.num_rounds; e++) {
    while (true) {
      hash_squeeze(&ctx, lambda_sized_buffer.data(),
                   lambda_sized_buffer.size());
      //  check that R is not in {0,...m2-1}
      field::GF2E candidate_R;
      candidate_R.from_bytes(lambda_sized_buffer.data());
      bool good = true;
      for (auto &fv: forbidden_values) {
        if (candidate_R == fv) {
          good = false;
          break;
        }
      }
      if (good) {
        R_es.push_back(candidate_R);
        break;
      }
    }
  }
  return R_es;
}

std::vector<uint8_t>
phase_3_commitment(const dubhe_instance_t &instance,
                   const dubhe_salt_t &salt, const std::vector<uint8_t> &h_2,
                   const std::vector<std::vector<field::GF2E>> &d_shares,
                   const std::vector<field::GF2E> &a,
                   const std::vector<std::vector<field::GF2E>> &a_shares,
                   const std::vector<field::GF2E> &b,
                   const std::vector<std::vector<field::GF2E>> &b_shares,
                   const std::vector<field::GF2E> &c,
                   const std::vector<std::vector<field::GF2E>> &c_shares) {

  hash_context ctx;
  hash_init_prefix(&ctx, instance.digest_size, HASH_PREFIX_3);
  hash_update(&ctx, salt.data(), salt.size());
  hash_update(&ctx, h_2.data(), h_2.size());

  for (size_t repetition = 0; repetition < instance.num_rounds; repetition++) {
    for (size_t party = 0; party < instance.num_MPC_parties; party++) {
      hash_update_GF2E(&ctx, instance, d_shares[repetition][party]);
    }
      hash_update_GF2E(&ctx, instance, a[repetition]);
      hash_update_GF2E(&ctx, instance, b[repetition]);
      hash_update_GF2E(&ctx, instance, c[repetition]);
      for (size_t party = 0; party < instance.num_MPC_parties; party++) {
        hash_update_GF2E(&ctx, instance, a_shares[repetition][party]);
        hash_update_GF2E(&ctx, instance, b_shares[repetition][party]);
        hash_update_GF2E(&ctx, instance, c_shares[repetition][party]);
      }
  }
  hash_final(&ctx);

  std::vector<uint8_t> commitment(instance.digest_size);
  hash_squeeze(&ctx, commitment.data(), commitment.size());
  return commitment;
}

std::vector<uint16_t> phase_3_expand(const dubhe_instance_t &instance,
                                     const std::vector<uint8_t> &h_3) {
  assert(instance.num_MPC_parties < (1ULL << 16));
  hash_context ctx;
  hash_init(&ctx, instance.digest_size);
  hash_update(&ctx, h_3.data(), h_3.size());
  hash_final(&ctx);
  size_t num_squeeze_bytes = instance.num_MPC_parties > 256 ? 2 : 1;

  std::vector<uint16_t> opened_parties;
  uint16_t mask = (1ULL << ceil_log2(instance.num_MPC_parties)) - 1;
  for (size_t e = 0; e < instance.num_rounds; e++) {
    uint16_t party;
    do {
      hash_squeeze(&ctx, (uint8_t *)&party, num_squeeze_bytes);
      party = le16toh(party);
      party = party & mask;
    } while (party >= instance.num_MPC_parties);
    opened_parties.push_back(party);
  }
  return opened_parties;
}
} // namespace

dubhe_keypair_t dubhe_keygen(const dubhe_instance_t &instance) {
    std::vector<uint8_t> key(instance.aes_params.key_size),
        pt(instance.aes_params.block_size * instance.aes_params.num_blocks),
        ct(instance.aes_params.block_size * instance.aes_params.num_blocks);

    rand_bytes(key.data(), key.size());
    rand_bytes(pt.data(), pt.size());
    if (instance.aes_params.key_size == 16) {
        AES128::aes_128(key, pt, ct);
    } else if (instance.aes_params.key_size == 24) {
        AES192::aes_192(key, pt, ct);
    } else if (instance.aes_params.key_size == 32) {
        AES256::aes_256(key, pt, ct);
    } else
        throw std::runtime_error("invalid parameters");

    dubhe_keypair_t keypair;
    keypair.first = key;
    keypair.second = pt;
    keypair.second.insert(keypair.second.end(), ct.begin(), ct.end());
    return keypair;
}

void gen_proof(const dubhe_instance_t &instance, const Circuit<field::GF2E> C, std::vector<uint8_t> inputs) {
// dubhe_signature_t dubhe_sign(const dubhe_instance_t &instance,
//                                  const dubhe_keypair_t &keypair,
//                                  const uint8_t *message, size_t message_len) {

    field::GF2E::init_extension_field(instance);

#ifdef TIMING
    timing_context_t ctx;
    timing_init(&ctx);

    uint64_t start_time = timing_read(&ctx);
    uint64_t tmp_time;
#endif

    // generate salt and master seeds for each repetition
    auto [salt, master_seeds] =
        generate_salt_and_seeds(instance);

    // do parallel repetitions
    // create seed trees and random tapes
    std::vector<SeedTree> seed_trees;
    seed_trees.reserve(instance.num_rounds);

    int bit_len = instance.aes_params.bit_len;
    // TODO: tape size
    size_t random_tape_size =
        instance.aes_params.key_size +
        instance.aes_params.num_sboxes +
        3 * bit_len * instance.lambda +
        2 * bit_len * instance.lambda +
        3 * instance.lambda + 3 * instance.lambda;

    RandomTapes random_tapes(instance.num_rounds, instance.num_MPC_parties,
            random_tape_size);

    RepByteContainer party_seed_commitments(
            instance.num_rounds, instance.num_MPC_parties, instance.digest_size);

    for (size_t repetition = 0; repetition < instance.num_rounds; repetition++) {
        // generate seed tree for the N parties
        seed_trees.emplace_back(master_seeds[repetition], instance.num_MPC_parties,
                salt, repetition);

        // commit to each party's seed;
        {
            size_t party = 0;
            for (; party < (instance.num_MPC_parties / 4) * 4; party += 4) {
                commit_to_4_party_seeds(
                        instance, seed_trees[repetition].get_leaf(party).value(),
                        seed_trees[repetition].get_leaf(party + 1).value(),
                        seed_trees[repetition].get_leaf(party + 2).value(),
                        seed_trees[repetition].get_leaf(party + 3).value(), salt,
                        repetition, party, party_seed_commitments.get(repetition, party),
                        party_seed_commitments.get(repetition, party + 1),
                        party_seed_commitments.get(repetition, party + 2),
                        party_seed_commitments.get(repetition, party + 3));
            }
            for (; party < instance.num_MPC_parties; party++) {
                commit_to_party_seed(
                        instance, seed_trees[repetition].get_leaf(party).value(), salt,
                        repetition, party, party_seed_commitments.get(repetition, party));
            }
    }

        {
            size_t party = 0;
            for (; party < (instance.num_MPC_parties / 4) * 4; party += 4) {
                random_tapes.generate_4_tapes(
                        repetition, party, salt,
                        seed_trees[repetition].get_leaf(party).value(),
                        seed_trees[repetition].get_leaf(party + 1).value(),
                        seed_trees[repetition].get_leaf(party + 2).value(),
                        seed_trees[repetition].get_leaf(party + 3).value());
            }
            for (; party < instance.num_MPC_parties; party++) {
                random_tapes.generate_tape(
                        repetition, party, salt,
                        seed_trees[repetition].get_leaf(party).value());
            }
        }
    }

    /////////////////////////////////////////////////////////////////////////////
    // phase 1: commit to inputs
    /////////////////////////////////////////////////////////////////////////////

    RepByteContainer rep_shared_input(instance.num_rounds,
            instance.num_MPC_parties,
            instance.aes_params.key_size);
    // RepByteContainer rep_output_broadcasts(
    //         instance.num_rounds, instance.num_MPC_parties,
    //         instance.aes_params.block_size * instance.aes_params.num_blocks);

    std::vector<std::vector<uint8_t>> rep_input_deltas;
    // std::vector<std::vector<uint8_t>> rep_t_deltas;

    for (size_t repetition = 0; repetition < instance.num_rounds; repetition++) {
        // generate sharing of secret key
        std::vector<uint8_t> input_delta = inputs;
        for (size_t party = 0; party < instance.num_MPC_parties; party++) {
            auto shared_input = rep_shared_input.get(repetition, party);
            auto random_input_share = random_tapes.get_bytes(repetition, party, 0, inputs.size());
            std::copy(std::begin(random_input_share), std::end(random_input_share),
                    std::begin(shared_input));

            std::transform(std::begin(shared_input), std::end(shared_input),
                    std::begin(input_delta), std::begin(input_delta),
                    std::bit_xor<uint8_t>());
        }

        // fix first share
        auto first_share_input = rep_shared_input.get(repetition, 0);
        std::transform(std::begin(input_delta), std::end(input_delta),
                std::begin(first_share_input), std::begin(first_share_input),
                   std::bit_xor<uint8_t>());

        rep_input_deltas.push_back(input_delta);
    }

    /////////////////////////////////////////////////////////////////////////////
    // phase 2: GKR
    /////////////////////////////////////////////////////////////////////////////


#ifdef TIMING
    tmp_time = timing_read(&ctx);
    printf("phase1 time: %ld\n", tmp_time - start_time);
    start_time = timing_read(&ctx);
#endif


    // commit to salt, (all commitments of parties seeds, key_delta, t_delta)
    // for all repetitions
    std::vector<uint8_t> h_1 = phase_1_commitment(instance, salt, party_seed_commitments, rep_input_deltas);

    std::vector<std::vector<field::GF2E>> tau = phase_sumcheck0_expand(instance, C, h_1);

    std::vector<uint8_t> h_i = h_1;

    RepContainer<field::GF2E> Vrs_shares(instance.num_rounds, instance.num_MPC_parties, 2);
    // for polynomials, a[0] = fr, b[0] = Vr, c[0] = Vs
    RepContainer<field::GF2E> fr_share(instance.num_rounds, instance.num_MPC_parties, 2);
    RepContainer<field::GF2E> Vr_share(instance.num_rounds, instance.num_MPC_parties, 2);
    RepContainer<field::GF2E> Vs_share(instance.num_rounds, instance.num_MPC_parties, 2);

    // sumcheck init
    std::vector<std::vector<uint16_t>> Tau(instance.num_rounds);
    std::vector<std::vector<uint16_t>> Rho(instance.num_rounds);
    std::vector<std::vector<uint16_t>> einputs(instance.num_rounds);

    // std::vector<std::vector<uint32_t>> Tau_v(instance.num_rounds);
    // std::vector<std::vector<uint32_t>> Rho_v(instance.num_rounds);
    // std::vector<std::vector<uint32_t>> inputs_v(instance.num_rounds);

    // std::vector<std::vector<std::vector<field::GF2E>>> inputs_rho_share(instance.num_rounds);
    // std::vector<std::vector<std::vector<field::GF2E>>> inputs_sgm_share(instance.num_rounds);

    std::vector<std::vector<std::vector<uint8_t>>> inputs_rho_share_v(instance.num_rounds);
    std::vector<std::vector<std::vector<uint8_t>>> inputs_sgm_share_v(instance.num_rounds);

    std::vector<std::vector<uint16_t>> AG(instance.num_rounds);
    std::vector<std::vector<uint16_t>> AX(instance.num_rounds);
    std::vector<std::vector<uint16_t>> AV(instance.num_rounds);

    // std::vector<std::vector<uint32_t>> AG_v(instance.num_rounds);
    // std::vector<std::vector<uint32_t>> AX_v(instance.num_rounds);
    // std::vector<std::vector<uint32_t>> AV_v(instance.num_rounds);

    std::vector<std::vector<std::array<field::GF2E, 3>>> coef1(bit_len);
    std::vector<std::vector<std::vector<field::GF2E>>> coef1_deltas;
    std::vector<std::vector<std::array<field::GF2E, 2>>> coef2(bit_len);
    std::vector<std::vector<std::vector<field::GF2E>>> coef2_deltas;
    // the second element is for the sum of each party's random
    std::vector<std::array<field::GF2E, 2>> Vr(instance.num_rounds);
    std::vector<std::array<field::GF2E, 2>> Vs(instance.num_rounds);
    std::vector<std::array<field::GF2E, 2>> fr(instance.num_rounds);
    std::vector<std::vector<field::GF2E>> rho;
    std::vector<field::GF2E> pred(instance.num_rounds, field::GF2E(1));

    // std::vector<std::vector<std::array<std::vector<field::GF2E>, 5>>> coef_shares(instance.num_rounds);
    // std::vector<std::vector<field::GF2E>> fr_shares(instance.num_rounds);

    std::vector<std::vector<std::array<std::vector<uint8_t>, 5>>> coef_shares_v(instance.num_rounds);
    std::vector<std::vector<uint8_t>> fr_shares_v(instance.num_rounds);

#ifdef TIMING
    tmp_time = timing_read(&ctx);
    printf("gkr init declare time: %ld\n", tmp_time - start_time);
    start_time = timing_read(&ctx);
#endif

    for (size_t e = 0; e < instance.num_rounds; e++) {
        // coef_shares[e].resize(bit_len);
        coef_shares_v[e].resize(bit_len);
        for (int i = 0; i < bit_len; i++)
            for (int j = 0; j < 5; j++) {
                // coef_shares[e][i][j].resize(instance.num_MPC_parties);
                coef_shares_v[e][i][j].resize(instance.num_MPC_parties * 2);
                // for (size_t p = 0; p < instance.num_MPC_parties * 2; p++)
                //     coef_shares_v[e][i][j][p] = 0;
            }
        // fr_shares[e].resize(instance.num_MPC_parties);
        fr_shares_v[e].resize(instance.num_MPC_parties * 2);
        // for (size_t p = 0; p < instance.num_MPC_parties * 2; p++)
        //     fr_shares_v[e][p] = 0;
    }

#ifdef TIMING
    tmp_time = timing_read(&ctx);
    printf("gkr init vectors time: %ld\n", tmp_time - start_time);
    start_time = timing_read(&ctx);
    uint64_t ttt;
    uint64_t init_time1 = 0;
    uint64_t init_time2 = 0;
#endif

    for (size_t e = 0; e < instance.num_rounds; e++) {
#ifdef TIMING
    ttt = timing_read(&ctx);
#endif
        // inputs[e].resize(1 << bit_len);
        inputs[e].resize(1 << bit_len);
        // inputs_rho_share[e].resize(1 << bit_len);
        // inputs_sgm_share[e].resize(1 << bit_len);

        inputs_rho_share_v[e].resize(1 << bit_len);
        inputs_sgm_share_v[e].resize(1 << bit_len);

        for (size_t i = 0; i < instance.aes_params.num_sboxes; i++) {
            int j = (1 << (bit_len - 1)) + i;
            inputs[e][i] = field::lift(sbox_pairs.first[i]);
            inputs[e][j] = field::lift(sbox_pairs.second[i]);
            // inputs[e][i] = field::GF2E(sbox_pairs.first[i]);
            // inputs[e][j] = field::GF2E(sbox_pairs.second[i]);
            // inputs_v[e][i] = sbox_pairs.first[i];
            // inputs_v[e][j] = sbox_pairs.second[i];
        }
        for (int i = 0; i < (1 << bit_len); i++) {
            // inputs_rho_share[e][i].resize(instance.num_MPC_parties);
            // inputs_sgm_share[e][i].resize(instance.num_MPC_parties);
            inputs_rho_share_v[e][i].resize(instance.num_MPC_parties * 2);
            inputs_sgm_share_v[e][i].resize(instance.num_MPC_parties * 2);
        }
#ifdef TIMING
    init_time1 += timing_read(&ctx) - ttt;
    ttt = timing_read(&ctx);
#endif
        // for (size_t p = 0; p < instance.num_MPC_parties; p++) {
        //     auto shared_s = rep_shared_s.get(e, p);
        //     auto shared_t = rep_shared_t.get(e, p);
        //     for (size_t i = 0; i < instance.aes_params.num_sboxes; i++) {
        //         size_t j = (1 << (bit_len - 1)) + i;
        //         inputs_rho_share[e][i][p] = field::lift_uint8_t(shared_s[i]);
        //         inputs_rho_share[e][j][p] = field::lift_uint8_t(shared_t[i]);
        //         inputs_sgm_share[e][i][p] = field::lift_uint8_t(shared_s[i]);
        //         inputs_sgm_share[e][j][p] = field::lift_uint8_t(shared_t[i]);
        //     }
        // }

        for (size_t i = 0; i < instance.aes_params.num_sboxes; i++) {
            size_t j = (1 << (bit_len - 1)) + i;
            for (size_t k = 0; k < instance.num_MPC_parties / 16; k++) {
                for (size_t p = 0; p < 16; p++) {
                    // auto si = rep_shared_s.get(e, k * 16 + p)[i];
                    // auto ti = rep_shared_t.get(e, k * 16 + p)[i];
                    // inputs_rho_share_v[e][i][k * 32 + p] = si;
                    // inputs_rho_share_v[e][j][k * 32 + p] = ti;
                    // inputs_sgm_share_v[e][i][k * 32 + p] = si;
                    // inputs_sgm_share_v[e][j][k * 32 + p] = ti;

                    auto si = field::lift_uint8_t(rep_shared_s.get(e, k * 16 + p)[i]).data;
                    auto ti = field::lift_uint8_t(rep_shared_t.get(e, k * 16 + p)[i]).data;
                    inputs_rho_share_v[e][i][k * 32 + p] = si & 255;
                    inputs_rho_share_v[e][i][k * 32 + 16 + p] = si >> 8;
                    inputs_rho_share_v[e][j][k * 32 + p] = ti & 255;
                    inputs_rho_share_v[e][j][k * 32 + 16 + p] = ti >> 8;
                    inputs_sgm_share_v[e][i][k * 32 + 16 + p] = si >> 8;
                    inputs_sgm_share_v[e][i][k * 32 + p] = si & 255;
                    inputs_sgm_share_v[e][j][k * 32 + 16 + p] = ti >> 8;
                    inputs_sgm_share_v[e][j][k * 32 + p] = ti & 255;
                }
            }
        }
#ifdef TIMING
    init_time2 += timing_read(&ctx) - ttt;
#endif
    }
    // AV = inputs;
    AV = inputs;

#ifdef TIMING
    tmp_time = timing_read(&ctx);
    printf("gkr init AV time: %ld\n", tmp_time - start_time);
    printf("gkr init AV time1: %ld\n", init_time1);
    printf("gkr init AV time2: %ld\n", init_time2);
    start_time = timing_read(&ctx);
#endif

    // sumcheck phase 1 init
    for (size_t e = 0; e < instance.num_rounds; e++) {
        // std::vector<field::GF2E> Tau_e(1 << bit_len);
        // Tau_e[0] = field::GF2E(1);
        // std::vector<field::GF2E> AX_e(1 << bit_len);
        // Rho[e].resize(1 << bit_len);
        // Rho[e][0] = field::GF2E(1);

        Tau[e].resize(1 << bit_len);
        Rho[e].resize(1 << bit_len);
        Tau[e][0] = 1;
        Rho[e][0] = 1;
        AX[e].resize(1 << bit_len);

        for (int k = 0; k < bit_len; k++) {
            int mask = 1 << k;
            for (int i = 0; i < mask; i++) {
                Tau[e][i + mask] = field::mul(Tau[e][i], tau[k][e].data);
                Tau[e][i] ^= Tau[e][i + mask];
            }
        }

        // for (int k = 0; k < bit_len; k++) {
        //     int mask = 1 << k;
        //     if (mask >= 8) {
        //         __m256i t_lo = _mm256_set1_epi32((uint32_t)(tau[k][e].data & 255) << 16);
        //         __m256i t_hi = _mm256_set1_epi32((256 + (uint32_t)(tau[k][e].data >> 8)) << 16);
        //         for (int i = 0; i < mask / 8; i++) {
        //             __m256i t_i = _mm256_loadu_si256((__m256i *)&(Tau_v[e][i * 8]));
        //             __m256i tlo_i  = _mm256_xor_si256(t_i, t_lo);
        //             __m256i thi_i  = _mm256_xor_si256(t_i, t_hi);
        //             tlo_i = _mm256_i32gather_epi32((const int *)F::mul_lut, tlo_i, 4);
        //             thi_i = _mm256_i32gather_epi32((const int *)F::mul_lut, thi_i, 4);
        //             __m256i t_j = _mm256_xor_si256(tlo_i, thi_i);
        //             _mm256_storeu_si256((__m256i *)&(Tau_v[e][i * 8]), _mm256_xor_si256(t_i, t_j));
        //             _mm256_storeu_si256((__m256i *)&(Tau_v[e][i * 8 + mask]), t_j);

        //             // if (k == bit_len - 1)
        //             // for (int j = 0; j < 8; j++) {
        //             //     if (Tau_e[i * 8 + j].data != Tau_v[e][i * 8 + j])
        //             //         printf("wtf0\n");
        //             //     if(Tau_e[i * 8 + j + mask].data != Tau_v[e][i * 8 + j + mask])
        //             //         printf("wtf1\n");
        //             // }
        //         }
        //     } else
        //         for (int i = 0; i < mask; i++) {
        //             Tau_v[e][i + mask] = (field::GF2E(Tau_v[e][i]) * tau[k][e]).data;
        //             Tau_v[e][i] ^= Tau_v[e][i + mask];
        //         }
        // }

        for (int i = 0; i < (1 << (bit_len - 1)); i++) {
            int j = (1 << (bit_len - 1)) + i;
            AX[e][i] = field::mul(Tau[e][i], AV[e][j]);
            AX[e][j] = field::mul(Tau[e][j], AV[e][i]);
        }

        // for (int i = 0; i < (1 << (bit_len - 1)); i += 8) {
        //     int j = (1 << (bit_len - 1)) + i;

        //     __m256i vi = _mm256_loadu_si256((__m256i *)&(AV_v[e][i]));
        //     __m256i ti = _mm256_loadu_si256((__m256i *)&(Tau_v[e][i]));
        //     __m256i vj = _mm256_loadu_si256((__m256i *)&(AV_v[e][j]));
        //     __m256i tj = _mm256_loadu_si256((__m256i *)&(Tau_v[e][j]));

        //     __m256i ti_lo = _mm256_slli_epi32(_mm256_and_si256(_mm256_set1_epi32(255), ti), 16);
        //     __m256i ti_hi = _mm256_or_si256(
        //             _mm256_slli_epi32(_mm256_and_si256(_mm256_set1_epi32(255 << 8), ti), 8),
        //             _mm256_set1_epi32(256 * 65536));

        //     ti_lo = _mm256_i32gather_epi32((const int *)F::mul_lut, _mm256_xor_si256(ti_lo, vj), 4);
        //     ti_hi = _mm256_i32gather_epi32((const int *)F::mul_lut, _mm256_xor_si256(ti_hi, vj), 4);

        //     __m256i tj_lo = _mm256_slli_epi32(_mm256_and_si256(_mm256_set1_epi32(255), tj), 16);
        //     __m256i tj_hi = _mm256_or_si256(
        //             _mm256_slli_epi32(_mm256_and_si256(_mm256_set1_epi32(255 << 8), tj), 8),
        //             _mm256_set1_epi32(256 * 65536));
        //     tj_lo = _mm256_i32gather_epi32((const int *)F::mul_lut, _mm256_xor_si256(tj_lo, vi), 4);
        //     tj_hi = _mm256_i32gather_epi32((const int *)F::mul_lut, _mm256_xor_si256(tj_hi, vi), 4);

        //     _mm256_storeu_si256((__m256i *)&(AX_v[e][i]), _mm256_xor_si256(ti_lo, ti_hi));
        //     _mm256_storeu_si256((__m256i *)&(AX_v[e][j]), _mm256_xor_si256(tj_lo, tj_hi));

        //     // for (int k; k < 8; k++) {
        //     //     if (AX_v[e][i + k] != AX_e[i + k].data)
        //     //         printf("wtf ax0\n");
        //     //     if (AX_v[e][j + k] != AX_e[j + k].data)
        //     //         printf("wtf ax1\n");
        //     // }
        // }

    }
    AG = Tau;

#ifdef TIMING
    tmp_time = timing_read(&ctx);
    printf("gkr phase 1 init time: %ld\n", tmp_time - start_time);
    start_time = timing_read(&ctx);

    uint64_t eval_time1 = 0;
    uint64_t eval_time2 = 0;
    uint64_t update_time1 = 0;
    uint64_t update_time2 = 0;
    uint64_t update_time3 = 0;
    uint64_t tt;
#endif

    // sumcheck phase 1 loop
    for (int i = 0; i < bit_len; i++) {
        coef1[i].resize(instance.num_rounds);
        std::vector<std::vector<field::GF2E>> coef_i_deltas;

#ifdef TIMING
        tt = timing_read(&ctx);
#endif
        // phase 1 evaluate
        for (size_t e = 0; e < instance.num_rounds; e++) {

            // field::GF2E tmp(0);
            for (int u = 0; u < (1 << (bit_len - i)); u += 2) {
                int one_u = u + 1;
                AV[e][one_u] ^= AV[e][u];
                AG[e][one_u] ^= AG[e][u];
                AX[e][one_u] ^= AX[e][u];
                // tmp += AV[e][u] * AV[e][u] * AX[e][one_u] + AV[e][u] * AG[e][one_u] + AG[e][u] * AV[e][one_u];

                uint16_t tmp0, tmp1;
                field::pair_mul(AX[e][u], AX[e][u],
                                AV[e][u], AV[e][one_u],
                                &tmp0, &tmp1);
                // field::pair_mul(AX[e][u], AX[e][u], AX[e][one_u],
                //                 AV[e][u], AV[e][one_u], AV[e][one_u],
                //                 &tmp0, &tmp1, &tmp2);
                field::pair_mul(tmp0 ^ AG[e][u], tmp1 ^ AG[e][one_u], AV[e][u], AV[e][one_u], &tmp0, &tmp1);
                // field::pair_mul(tmp0 ^ AG[e][u], tmp1 ^ AG[e][one_u], tmp2,
                //                 AV[e][u], AV[e][one_u], AV[e][one_u],
                //                 &tmp0, &tmp1, &tmp2);
                coef1[i][e][0] += field::GF2E(tmp0);
                coef1[i][e][1] += field::GF2E(tmp1);
                // coef1[i][e][2] += field::GF2E(tmp2);
                coef1[i][e][2] += field::GF2E(field::mul(AV[e][one_u], field::mul(AV[e][one_u], AX[e][one_u])));

                // coef1[i][e][0] += AV[e][u] * (AV[e][u] * AX[e][u] + AG[e][u]);
                // coef1[i][e][1] += AV[e][one_u] * (AV[e][one_u] * AX[e][u] + AG[e][one_u]);
                // coef1[i][e][2] += field::GF2E(field::mul(AV[e][one_u], field::mul(AV[e][one_u], AX[e][one_u])));
            }

            // for (int u = 0; u < (1 << (bit_len - i)); u += 2) {
            //     int one_u = u + 1;
            //     // AH[e][one_u] -= AH[e][u];   // all zero
            //     AV[e][one_u] -= AV[e][u];
            //     AG[e][one_u] -= AG[e][u];
            //     AX[e][one_u] -= AX[e][u];
            //     // if (AG_v[e][u] != AG[e][u].data)
            //     //     printf("wtf\n");
            //     // tmp += AV[e][u] * AV[e][u] * AX[e][one_u] + AV[e][u] * AG[e][one_u] + AG[e][u] * AV[e][one_u];
            //     coef1[i][e][0] += AV[e][u] * AV[e][u] * AX[e][u] + AG[e][u];
            //     coef1[i][e][1] += AV[e][one_u] * (AV[e][one_u] * AX[e][u] + AG[e][one_u]);
            //     coef1[i][e][2] += AV[e][one_u] * AV[e][one_u] * AX[e][one_u];
            // }

            fr[e][0] -= coef1[i][e][1] + coef1[i][e][2];
            // if (tmp != fr[e][0])
            //     printf("phase 1 wtffff %d\n", i);
        }

#ifdef TIMING
        eval_time1 += timing_read(&ctx) - tt;
        tt = timing_read(&ctx);
#endif

        for (size_t e = 0; e < instance.num_rounds; e++) {
            std::vector<field::GF2E> coef_ie_deltas(3);

            __m256i coef_ie_deltas_v[3];
            for (int k = 0; k < 3; k++)
                coef_ie_deltas_v[k] = _mm256_setzero_si256();

            for (size_t k = 0; k < instance.num_MPC_parties / 16; k++) {
                for (size_t p = 0; p < 16; p++) {
                    auto random_coef_share =
                        random_tapes.get_bytes(e, k * 16 + p,
                                instance.aes_params.key_size + instance.aes_params.num_sboxes + 3 * i * instance.lambda,
                                3 * instance.lambda);
                    coef_shares_v[e][i][0][k * 32 + p] = *(uint8_t *)(random_coef_share.data());
                    coef_shares_v[e][i][1][k * 32 + p] = *(uint8_t *)(random_coef_share.data() + 2);
                    coef_shares_v[e][i][2][k * 32 + p] = *(uint8_t *)(random_coef_share.data() + 4);
                    coef_shares_v[e][i][0][k * 32 + 16 + p] = *(uint8_t *)(random_coef_share.data() + 1);
                    coef_shares_v[e][i][1][k * 32 + 16 + p] = *(uint8_t *)(random_coef_share.data() + 3);
                    coef_shares_v[e][i][2][k * 32 + 16 + p] = *(uint8_t *)(random_coef_share.data() + 5);
                }
                __m256i t_coef0 = _mm256_loadu_si256((__m256i *)&(coef_shares_v[e][i][0][k * 32]));
                __m256i t_coef1 = _mm256_loadu_si256((__m256i *)&(coef_shares_v[e][i][1][k * 32]));
                __m256i t_coef2 = _mm256_loadu_si256((__m256i *)&(coef_shares_v[e][i][2][k * 32]));
                __m256i t_fr    = _mm256_loadu_si256((__m256i *)&(fr_shares_v[e][k * 32]));
                t_fr = _mm256_xor_si256(_mm256_xor_si256(t_coef1, t_coef2), t_fr);
                _mm256_storeu_si256((__m256i *)&(fr_shares_v[e][k * 32]), t_fr);
                coef_ie_deltas_v[0] = _mm256_xor_si256(coef_ie_deltas_v[0], t_coef0);
                coef_ie_deltas_v[1] = _mm256_xor_si256(coef_ie_deltas_v[1], t_coef1);
                coef_ie_deltas_v[2] = _mm256_xor_si256(coef_ie_deltas_v[2], t_coef2);
            }
            coef_ie_deltas[0] = coef1[i][e][0] - field::GF2E(_mm256_hxor_epu16(coef_ie_deltas_v[0]));
            coef_ie_deltas[1] = coef1[i][e][1] - field::GF2E(_mm256_hxor_epu16(coef_ie_deltas_v[1]));
            coef_ie_deltas[2] = coef1[i][e][2] - field::GF2E(_mm256_hxor_epu16(coef_ie_deltas_v[2]));
            coef_i_deltas.push_back(coef_ie_deltas);

            coef_shares_v[e][i][0][0] ^= (coef_ie_deltas[0].data & 255);
            coef_shares_v[e][i][1][0] ^= (coef_ie_deltas[1].data & 255);
            coef_shares_v[e][i][2][0] ^= (coef_ie_deltas[2].data & 255);
            coef_shares_v[e][i][0][16] ^= (coef_ie_deltas[0].data >> 8);
            coef_shares_v[e][i][1][16] ^= (coef_ie_deltas[1].data >> 8);
            coef_shares_v[e][i][2][16] ^= (coef_ie_deltas[2].data >> 8);

            fr_shares_v[e][0] ^= (coef_ie_deltas[1] + coef_ie_deltas[2]).data & 255;
            fr_shares_v[e][16] ^= (coef_ie_deltas[1] + coef_ie_deltas[2]).data >> 8;

            // for (size_t p = 0; p < instance.num_MPC_parties; p++) {
            //     auto random_coef_share =
            //         random_tapes.get_bytes(e, p,
            //                 instance.aes_params.key_size + instance.aes_params.num_sboxes + 3 * i * instance.lambda,
            //                 3 * instance.lambda);
            //     coef_shares[e][i][0][p].from_bytes(random_coef_share.data());
            //     coef_shares[e][i][1][p].from_bytes(random_coef_share.data() + instance.lambda);
            //     coef_shares[e][i][2][p].from_bytes(random_coef_share.data() + instance.lambda * 2);

            //     fr_shares[e][p] -= coef_shares[e][i][1][p] + coef_shares[e][i][2][p];
            //     coef_ie_deltas[0] += coef_shares[e][i][0][p];
            //     coef_ie_deltas[1] += coef_shares[e][i][1][p];
            //     coef_ie_deltas[2] += coef_shares[e][i][2][p];
            // }
            // coef_ie_deltas[0] = coef1[i][e][0] - coef_ie_deltas[0];
            // coef_ie_deltas[1] = coef1[i][e][1] - coef_ie_deltas[1];
            // coef_ie_deltas[2] = coef1[i][e][2] - coef_ie_deltas[2];
            // coef_i_deltas.push_back(coef_ie_deltas);
            // coef_shares[e][i][0][0] += coef_ie_deltas[0];
            // coef_shares[e][i][1][0] += coef_ie_deltas[1];
            // coef_shares[e][i][2][0] += coef_ie_deltas[2];
            // fr_shares[e][0] += coef_ie_deltas[1] + coef_ie_deltas[2];

            // if ((fr_shares[e][1].data & 255) != fr_shares_v[e][1])
            //     printf("wtf\n");
            // if ((fr_shares[e][1].data >> 8) != fr_shares_v[e][17])
            //     printf("wtf2\n");

        }
        coef1_deltas.push_back(coef_i_deltas);

#ifdef TIMING
        eval_time2 += timing_read(&ctx) - tt;
        tt = timing_read(&ctx);
#endif

        h_i =
            // phase_gkr_commitment(instance, salt, keypair.second, message, message_len,
            //                    party_seed_commitments, rep_key_deltas, rep_t_deltas,
            //                    rep_output_broadcasts);
            phase_sumcheck_commitment(instance, salt, h_i, coef1_deltas[i]);

        std::vector<field::GF2E> rho_i = phase_sumcheck_expand(instance, h_i);
        rho.push_back(rho_i);

        for (size_t e = 0; e < instance.num_rounds; e++) {
            //
            auto r = rho_i[e];
            auto r_raw = r.data;
            // __m256i r_lo    = _mm256_set1_epi32((uint32_t)(r.data & 255) << 16);
            // __m256i r_hi    = _mm256_set1_epi32((256 + (uint32_t)(r.data >> 8)) << 16);
            fr[e][0] = ((coef1[i][e][2] * r + coef1[i][e][1]) * r + fr[e][0]) * r + coef1[i][e][0];
            // for (size_t p = 0; p < instance.num_MPC_parties; p++) {
            //     fr_shares[e][p] = ((coef_shares[e][i][2][p] * r + coef_shares[e][i][1][p]) * r + fr_shares[e][p]) * r + coef_shares[e][i][0][p];
            // }

#ifdef TIMING
        tt = timing_read(&ctx);
#endif
            const __m256i table0 = _mm256_loadu_si256((__m256i *)&(field::GF2E::mul_lut[r_raw][0]));
            const __m256i table1 = _mm256_loadu_si256((__m256i *)&(field::GF2E::mul_lut[r_raw][8]));
            const __m256i table2 = _mm256_loadu_si256((__m256i *)&(field::GF2E::mul_lut[r_raw][16]));
            const __m256i table3 = _mm256_loadu_si256((__m256i *)&(field::GF2E::mul_lut[r_raw][24]));
            const __m256i mask1 = _mm256_set1_epi8(0x0f);
            const __m256i mask2 = _mm256_set1_epi8(0xf0);
            // printf("r = %04lx\n", r_raw);
            // printf("table0: ");
            // print(table0);
            for (size_t k = 0; k < instance.num_MPC_parties / 16; k++) {
                __m256i lo, hi, tmp;
                __m256i t_coef0 = _mm256_loadu_si256((__m256i *)&(coef_shares_v[e][i][0][k * 32]));
                __m256i t_coef1 = _mm256_loadu_si256((__m256i *)&(coef_shares_v[e][i][1][k * 32]));
                __m256i t_coef2 = _mm256_loadu_si256((__m256i *)&(coef_shares_v[e][i][2][k * 32]));
                __m256i t_fr    = _mm256_loadu_si256((__m256i *)&(fr_shares_v[e][k * 32]));

                // coef[2] * r
                //
                lo = _mm256_and_si256(t_coef2, mask1);  // a0, a2
                lo = _mm256_shuffle_epi8(table0, lo);   // a0lo, a2hi
                hi = _mm256_srli_epi64(_mm256_and_si256(t_coef2, mask2), 4);    // a1, a3
                hi = _mm256_shuffle_epi8(table1, hi);   // a1lo, a3hi
                tmp = _mm256_xor_si256(hi, lo);
                t_coef2 = _mm256_permute4x64_epi64(t_coef2, 2 | (3 << 2) | (1 << 6));   // a2,a3; a0,a1
                lo = _mm256_and_si256(t_coef2, mask1);  // a2, a0
                lo = _mm256_shuffle_epi8(table2, lo);   // a2lo, a0hi
                hi = _mm256_srli_epi64(_mm256_and_si256(t_coef2, mask2), 4);    // a3, a1
                hi = _mm256_shuffle_epi8(table3, hi);   // a3lo, a1hi
                t_coef2 = _mm256_xor_si256(tmp, _mm256_xor_si256(hi, lo));

                // + coef[1]
                t_coef2 = _mm256_xor_si256(t_coef1, t_coef2);
                // * r
                lo = _mm256_and_si256(t_coef2, mask1);  // a0, a2
                lo = _mm256_shuffle_epi8(table0, lo);   // a0lo, a2hi
                hi = _mm256_srli_epi64(_mm256_and_si256(t_coef2, mask2), 4);    // a1, a3
                hi = _mm256_shuffle_epi8(table1, hi);   // a1lo, a3hi
                tmp = _mm256_xor_si256(hi, lo);
                t_coef2 = _mm256_permute4x64_epi64(t_coef2, 2 | (3 << 2) | (1 << 6));   // a2,a3; a0,a1
                lo = _mm256_and_si256(t_coef2, mask1);  // a2, a0
                lo = _mm256_shuffle_epi8(table2, lo);   // a2lo, a0hi
                hi = _mm256_srli_epi64(_mm256_and_si256(t_coef2, mask2), 4);    // a3, a1
                hi = _mm256_shuffle_epi8(table3, hi);   // a3lo, a1hi
                t_coef2 = _mm256_xor_si256(tmp, _mm256_xor_si256(hi, lo));
                // + fr
                t_coef2 = _mm256_xor_si256(t_fr, t_coef2);
                // * r
                lo = _mm256_and_si256(t_coef2, mask1);  // a0, a2
                lo = _mm256_shuffle_epi8(table0, lo);   // a0lo, a2hi
                hi = _mm256_srli_epi64(_mm256_and_si256(t_coef2, mask2), 4);    // a1, a3
                hi = _mm256_shuffle_epi8(table1, hi);   // a1lo, a3hi
                tmp = _mm256_xor_si256(hi, lo);
                t_coef2 = _mm256_permute4x64_epi64(t_coef2, 2 | (3 << 2) | (1 << 6));   // a2,a3; a0,a1
                lo = _mm256_and_si256(t_coef2, mask1);  // a2, a0
                lo = _mm256_shuffle_epi8(table2, lo);   // a2lo, a0hi
                hi = _mm256_srli_epi64(_mm256_and_si256(t_coef2, mask2), 4);    // a3, a1
                hi = _mm256_shuffle_epi8(table3, hi);   // a3lo, a1hi
                t_coef2 = _mm256_xor_si256(tmp, _mm256_xor_si256(hi, lo));
                // + coef[0]
                t_coef2 = _mm256_xor_si256(t_coef0, t_coef2);

                _mm256_storeu_si256((__m256i *)&(fr_shares_v[e][k * 32]), t_coef2);

                // for (int p = 0; p < 16; p++) {
                //     if (fr_shares_v[e][k * 32 + p] != (fr_shares[e][k * 16 + p].data & 255))
                //         printf("wtf0\n");
                //     if (fr_shares_v[e][k * 32 + p + 16] != (fr_shares[e][k * 16 + p].data >> 8))
                //         printf("wtf1\n");
                // }
            }

            // for (size_t p = 0; p < instance.num_MPC_parties / 8; p++) {
            //     __m256i t_coef0 = _mm256_loadu_si256((__m256i *)&(coef_shares_v[e][i][0][p * 8]));
            //     __m256i t_coef1 = _mm256_loadu_si256((__m256i *)&(coef_shares_v[e][i][1][p * 8]));
            //     __m256i t_coef2 = _mm256_loadu_si256((__m256i *)&(coef_shares_v[e][i][2][p * 8]));
            //     __m256i t_fr    = _mm256_loadu_si256((__m256i *)&(fr_shares_v[e][p * 8]));

            //     __m256i rlo_c  = _mm256_xor_si256(t_coef2, r_lo);
            //     __m256i rhi_c  = _mm256_xor_si256(t_coef2, r_hi);
            //     rlo_c = _mm256_i32gather_epi32((const int *)F::mul_lut, rlo_c, 4);
            //     rhi_c = _mm256_i32gather_epi32((const int *)F::mul_lut, rhi_c, 4);
            //     t_coef1 = _mm256_xor_si256(t_coef1, _mm256_xor_si256(rlo_c, rhi_c));

            //     rlo_c  = _mm256_xor_si256(t_coef1, r_lo);
            //     rhi_c  = _mm256_xor_si256(t_coef1, r_hi);
            //     rlo_c = _mm256_i32gather_epi32((const int *)F::mul_lut, rlo_c, 4);
            //     rhi_c = _mm256_i32gather_epi32((const int *)F::mul_lut, rhi_c, 4);
            //     t_fr = _mm256_xor_si256(t_fr, _mm256_xor_si256(rlo_c, rhi_c));

            //     rlo_c  = _mm256_xor_si256(t_fr, r_lo);
            //     rhi_c  = _mm256_xor_si256(t_fr, r_hi);
            //     rlo_c = _mm256_i32gather_epi32((const int *)F::mul_lut, rlo_c, 4);
            //     rhi_c = _mm256_i32gather_epi32((const int *)F::mul_lut, rhi_c, 4);
            //     t_fr = _mm256_xor_si256(t_coef0, _mm256_xor_si256(rlo_c, rhi_c));
            //     _mm256_storeu_si256((__m256i *)&(fr_shares_v[e][p * 8]), t_fr);

            //     // for (int k = 0; k < 8; k++)
            //     //     if (fr_shares_v[e][p * 8 + k] != fr_shares[e][p * 8 + k].data)
            //     //         printf("wtf\n");
            // }

#ifdef TIMING
        update_time2 += timing_read(&ctx) - tt;
        tt = timing_read(&ctx);
#endif

            // phase 1 update
            int mask = 1 << (bit_len - i - 1);
            // if (mask < 16)
                for (int u = 0; u < mask; u++) {
                    int u0 = u << 1;
                    int u1 = u0 + 1;
                    // for (size_t p = 0; p < instance.num_MPC_parties; p++)
                    //     inputs_rho_share[e][u][p] = inputs_rho_share[e][u0][p] + (inputs_rho_share[e][u1][p] - inputs_rho_share[e][u0][p]) * r;
                    for (size_t k = 0; k < instance.num_MPC_parties / 16; k++) {
                        __m256i lo, hi, tmp;
                        __m256i t_u0 = _mm256_loadu_si256((__m256i *)&(inputs_rho_share_v[e][u0][k * 32]));
                        __m256i t_u1 = _mm256_loadu_si256((__m256i *)&(inputs_rho_share_v[e][u1][k * 32]));
                        t_u1 = _mm256_xor_si256(t_u0, t_u1);

                        // (t_u1 + t_u0) * r
                        lo = _mm256_and_si256(t_u1, mask1);  // a0, a2
                        lo = _mm256_shuffle_epi8(table0, lo);   // a0lo, a2hi
                        hi = _mm256_srli_epi64(_mm256_and_si256(t_u1, mask2), 4);    // a1, a3
                        hi = _mm256_shuffle_epi8(table1, hi);   // a1lo, a3hi
                        tmp = _mm256_xor_si256(hi, lo);
                        t_u1 = _mm256_permute4x64_epi64(t_u1, 2 | (3 << 2) | (1 << 6));   // a2,a3; a0,a1
                        lo = _mm256_and_si256(t_u1, mask1);  // a2, a0
                        lo = _mm256_shuffle_epi8(table2, lo);   // a2lo, a0hi
                        hi = _mm256_srli_epi64(_mm256_and_si256(t_u1, mask2), 4);    // a3, a1
                        hi = _mm256_shuffle_epi8(table3, hi);   // a3lo, a1hi
                        t_u1 = _mm256_xor_si256(tmp, _mm256_xor_si256(hi, lo));

                        // + t_u0
                        t_u1 = _mm256_xor_si256(t_u0, t_u1);
                        _mm256_storeu_si256((__m256i *)&(inputs_rho_share_v[e][u][k * 32]), t_u1);
                    }

                    field::scale_mul(AV[e][u1], AG[e][u1], AX[e][u1], r.data,
                            AV[e][u0], AG[e][u0], AX[e][u0],
                            &AV[e][u], &AG[e][u], &AX[e][u]);
                    // AV[e][u] = AV[e][u0] ^ AV[e][u1];
                    // AG[e][u] = AG[e][u0] ^ AG[e][u1];
                    // AX[e][u] = AX[e][u0] ^ field::mul(AX[e][u1], r.data);
                }
            // else
            //     for (int uu = 0; uu < mask/16; uu++) {
            //         for (int u = 0; u < 16; u++) {
            //             int u0 = (uu * 16 + u) << 1;
            //             int u1 = u0 + 1;
            //             // for (size_t p = 0; p < instance.num_MPC_parties; p++)
            //             //     inputs_rho_share[e][u][p] = inputs_rho_share[e][u0][p] + (inputs_rho_share[e][u1][p] - inputs_rho_share[e][u0][p]) * r;
            //             for (size_t k = 0; k < instance.num_MPC_parties / 16; k++) {
            //                 __m256i lo, hi, tmp;
            //                 __m256i t_u0 = _mm256_loadu_si256((__m256i *)&(inputs_rho_share_v[e][u0][k * 32]));
            //                 __m256i t_u1 = _mm256_loadu_si256((__m256i *)&(inputs_rho_share_v[e][u1][k * 32]));
            //                 t_u1 = _mm256_xor_si256(t_u0, t_u1);

            //                 // (t_u1 + t_u0) * r
            //                 lo = _mm256_and_si256(t_u1, mask1);  // a0, a2
            //                 lo = _mm256_shuffle_epi8(table0, lo);   // a0lo, a2hi
            //                 hi = _mm256_srli_epi64(_mm256_and_si256(t_u1, mask2), 4);    // a1, a3
            //                 hi = _mm256_shuffle_epi8(table1, hi);   // a1lo, a3hi
            //                 tmp = _mm256_xor_si256(hi, lo);
            //                 t_u1 = _mm256_permute4x64_epi64(t_u1, 2 | (3 << 2) | (1 << 6));   // a2,a3; a0,a1
            //                 lo = _mm256_and_si256(t_u1, mask1);  // a2, a0
            //                 lo = _mm256_shuffle_epi8(table2, lo);   // a2lo, a0hi
            //                 hi = _mm256_srli_epi64(_mm256_and_si256(t_u1, mask2), 4);    // a3, a1
            //                 hi = _mm256_shuffle_epi8(table3, hi);   // a3lo, a1hi
            //                 t_u1 = _mm256_xor_si256(tmp, _mm256_xor_si256(hi, lo));

            //                 // + t_u0
            //                 t_u1 = _mm256_xor_si256(t_u0, t_u1);
            //                 _mm256_storeu_si256((__m256i *)&(inputs_rho_share_v[e][uu * 16 + u][k * 32]), t_u1);
            //             }
            //         }

            //         __m256i lo, hi, tmp;
            //         __m256i s_u0, s_u1, v_u0, v_u1;
            //         // AV
            //         s_u0 = _mm256_loadu_si256((__m256i *)&(AV[e][uu * 16 * 2]));
            //         s_u1 = _mm256_loadu_si256((__m256i *)&(AV[e][uu * 16 * 2 + 16]));
            //         v_u0 = _mm256_packus_epi32(
            //                 _mm256_and_si256(s_u0, _mm256_set1_epi32(0xffff)),
            //                 _mm256_and_si256(s_u1, _mm256_set1_epi32(0xffff)));
            //         v_u0 = _mm256_permute4x64_epi64(v_u0, 0b11011000);
            //         v_u1 = _mm256_packus_epi32(
            //                 _mm256_srli_epi32(s_u0, 16),
            //                 _mm256_srli_epi32(s_u1, 16));
            //         v_u1 = _mm256_permute4x64_epi64(v_u1, 0b11011000);
            //         // to alt mapping
            //         lo = _mm256_packus_epi16(_mm256_and_si256(v_u1, _mm256_set1_epi16(0xff)), _mm256_set1_epi64x(0));
            //         lo = _mm256_permute4x64_epi64(lo, 0b11011000);
            //         hi = _mm256_packus_epi16(_mm256_set1_epi64x(0), _mm256_srli_epi16(v_u1, 8));
            //         hi = _mm256_permute4x64_epi64(hi, 0b11011000);
            //         v_u1 = _mm256_xor_si256(lo, hi);

            //         // u1 * r
            //         lo = _mm256_and_si256(v_u1, mask1);  // a0, a2
            //         lo = _mm256_shuffle_epi8(table0, lo);   // a0lo, a2hi
            //         hi = _mm256_srli_epi64(_mm256_and_si256(v_u1, mask2), 4);    // a1, a3
            //         hi = _mm256_shuffle_epi8(table1, hi);   // a1lo, a3hi
            //         tmp = _mm256_xor_si256(hi, lo);
            //         v_u1 = _mm256_permute4x64_epi64(v_u1, 2 | (3 << 2) | (1 << 6));   // a2,a3; a0,a1
            //         lo = _mm256_and_si256(v_u1, mask1);  // a2, a0
            //         lo = _mm256_shuffle_epi8(table2, lo);   // a2lo, a0hi
            //         hi = _mm256_srli_epi64(_mm256_and_si256(v_u1, mask2), 4);    // a3, a1
            //         hi = _mm256_shuffle_epi8(table3, hi);   // a3lo, a1hi
            //         v_u1 = _mm256_xor_si256(tmp, _mm256_xor_si256(hi, lo));

            //         // to standard mapping
            //         v_u1 = _mm256_unpacklo_epi8(
            //                 _mm256_permute4x64_epi64(v_u1, 0b11011000),
            //                 _mm256_permute4x64_epi64(v_u1, 0b01110010));

            //         v_u1 = _mm256_xor_si256(v_u0, v_u1);

            //         _mm256_storeu_si256((__m256i *)&(AV[e][uu * 16]), v_u1);

            //         // AG
            //         s_u0 = _mm256_loadu_si256((__m256i *)&(AG[e][uu * 16 * 2]));
            //         s_u1 = _mm256_loadu_si256((__m256i *)&(AG[e][uu * 16 * 2 + 16]));
            //         v_u0 = _mm256_packus_epi32(
            //                 _mm256_and_si256(s_u0, _mm256_set1_epi32(0xffff)),
            //                 _mm256_and_si256(s_u1, _mm256_set1_epi32(0xffff)));
            //         v_u0 = _mm256_permute4x64_epi64(v_u0, 0b11011000);
            //         v_u1 = _mm256_packus_epi32(
            //                 _mm256_srli_epi32(s_u0, 16),
            //                 _mm256_srli_epi32(s_u1, 16));
            //         v_u1 = _mm256_permute4x64_epi64(v_u1, 0b11011000);
            //         // to alt mapping
            //         lo = _mm256_packus_epi16(_mm256_and_si256(v_u1, _mm256_set1_epi16(0xff)), _mm256_set1_epi64x(0));
            //         lo = _mm256_permute4x64_epi64(lo, 0b11011000);
            //         hi = _mm256_packus_epi16(_mm256_set1_epi64x(0), _mm256_srli_epi16(v_u1, 8));
            //         hi = _mm256_permute4x64_epi64(hi, 0b11011000);
            //         v_u1 = _mm256_xor_si256(lo, hi);

            //         // u1 * r
            //         lo = _mm256_and_si256(v_u1, mask1);  // a0, a2
            //         lo = _mm256_shuffle_epi8(table0, lo);   // a0lo, a2hi
            //         hi = _mm256_srli_epi64(_mm256_and_si256(v_u1, mask2), 4);    // a1, a3
            //         hi = _mm256_shuffle_epi8(table1, hi);   // a1lo, a3hi
            //         tmp = _mm256_xor_si256(hi, lo);
            //         v_u1 = _mm256_permute4x64_epi64(v_u1, 2 | (3 << 2) | (1 << 6));   // a2,a3; a0,a1
            //         lo = _mm256_and_si256(v_u1, mask1);  // a2, a0
            //         lo = _mm256_shuffle_epi8(table2, lo);   // a2lo, a0hi
            //         hi = _mm256_srli_epi64(_mm256_and_si256(v_u1, mask2), 4);    // a3, a1
            //         hi = _mm256_shuffle_epi8(table3, hi);   // a3lo, a1hi
            //         v_u1 = _mm256_xor_si256(tmp, _mm256_xor_si256(hi, lo));

            //         // to standard mapping
            //         v_u1 = _mm256_unpacklo_epi8(
            //                 _mm256_permute4x64_epi64(v_u1, 0b11011000),
            //                 _mm256_permute4x64_epi64(v_u1, 0b01110010));

            //         v_u1 = _mm256_xor_si256(v_u0, v_u1);

            //         _mm256_storeu_si256((__m256i *)&(AG[e][uu * 16]), v_u1);

            //         // AX
            //         s_u0 = _mm256_loadu_si256((__m256i *)&(AX[e][uu * 16 * 2]));
            //         s_u1 = _mm256_loadu_si256((__m256i *)&(AX[e][uu * 16 * 2 + 16]));
            //         v_u0 = _mm256_packus_epi32(
            //                 _mm256_and_si256(s_u0, _mm256_set1_epi32(0xffff)),
            //                 _mm256_and_si256(s_u1, _mm256_set1_epi32(0xffff)));
            //         v_u0 = _mm256_permute4x64_epi64(v_u0, 0b11011000);
            //         v_u1 = _mm256_packus_epi32(
            //                 _mm256_srli_epi32(s_u0, 16),
            //                 _mm256_srli_epi32(s_u1, 16));
            //         v_u1 = _mm256_permute4x64_epi64(v_u1, 0b11011000);
            //         // to alt mapping
            //         lo = _mm256_packus_epi16(_mm256_and_si256(v_u1, _mm256_set1_epi16(0xff)), _mm256_set1_epi64x(0));
            //         lo = _mm256_permute4x64_epi64(lo, 0b11011000);
            //         hi = _mm256_packus_epi16(_mm256_set1_epi64x(0), _mm256_srli_epi16(v_u1, 8));
            //         hi = _mm256_permute4x64_epi64(hi, 0b11011000);
            //         v_u1 = _mm256_xor_si256(lo, hi);

            //         // u1 * r
            //         lo = _mm256_and_si256(v_u1, mask1);  // a0, a2
            //         lo = _mm256_shuffle_epi8(table0, lo);   // a0lo, a2hi
            //         hi = _mm256_srli_epi64(_mm256_and_si256(v_u1, mask2), 4);    // a1, a3
            //         hi = _mm256_shuffle_epi8(table1, hi);   // a1lo, a3hi
            //         tmp = _mm256_xor_si256(hi, lo);
            //         v_u1 = _mm256_permute4x64_epi64(v_u1, 2 | (3 << 2) | (1 << 6));   // a2,a3; a0,a1
            //         lo = _mm256_and_si256(v_u1, mask1);  // a2, a0
            //         lo = _mm256_shuffle_epi8(table2, lo);   // a2lo, a0hi
            //         hi = _mm256_srli_epi64(_mm256_and_si256(v_u1, mask2), 4);    // a3, a1
            //         hi = _mm256_shuffle_epi8(table3, hi);   // a3lo, a1hi
            //         v_u1 = _mm256_xor_si256(tmp, _mm256_xor_si256(hi, lo));

            //         // to standard mapping
            //         v_u1 = _mm256_unpacklo_epi8(
            //                 _mm256_permute4x64_epi64(v_u1, 0b11011000),
            //                 _mm256_permute4x64_epi64(v_u1, 0b01110010));

            //         v_u1 = _mm256_xor_si256(v_u0, v_u1);

            //         _mm256_storeu_si256((__m256i *)&(AX[e][uu * 16]), v_u1);

            //         // for (int u = 0; u < 16; u++) {
            //         //     int u0 = (uu * 16 + u) << 1;
            //         //     int u1 = u0 + 1;
            //         //     AV[e][uu * 16 + u] = AV[e][u0] ^ field::mul(AV[e][u1], r.data);
            //         //     AG[e][uu * 16 + u] = AG[e][u0] ^ field::mul(AG[e][u1], r.data);
            //         //     AX[e][uu * 16 + u] = AX[e][u0] ^ field::mul(AX[e][u1], r.data);
            //         // }
            //     }

            // if (mask < 8)
            //     for (int u = 0; u < mask; u++) {
            //         int u0 = u << 1;
            //         int u1 = u0 + 1;
            //         for (size_t p = 0; p < instance.num_MPC_parties / 8; p++) {
            //             __m256i t_u0 = _mm256_loadu_si256((__m256i *)&(inputs_rho_share_v[e][u0][p * 8]));
            //             __m256i t_u1 = _mm256_loadu_si256((__m256i *)&(inputs_rho_share_v[e][u1][p * 8]));
            //             t_u1 = _mm256_xor_si256(t_u1, t_u0);
            //             __m256i rlo_c  = _mm256_xor_si256(t_u1, r_lo);
            //             __m256i rhi_c  = _mm256_xor_si256(t_u1, r_hi);
            //             rlo_c = _mm256_i32gather_epi32((const int *)F::mul_lut, rlo_c, 4);
            //             rhi_c = _mm256_i32gather_epi32((const int *)F::mul_lut, rhi_c, 4);
            //             t_u1 = _mm256_xor_si256(t_u0, _mm256_xor_si256(rlo_c, rhi_c));
            //             _mm256_storeu_si256((__m256i *)&(inputs_rho_share_v[e][u][p * 8]), t_u1);
            //         }
            //         AV[e][u] = AV[e][u0] + AV[e][u1] * r;
            //         AG[e][u] = AG[e][u0] + AG[e][u1] * r;
            //         AX[e][u] = AX[e][u0] + AX[e][u1] * r;
            //     }
            // else
            //     for (int uu = 0; uu < mask; uu += 8) {
            //         for (int u = uu; u < uu + 8; u++) {
            //             int u0 = u << 1;
            //             int u1 = u0 + 1;
            //             for (size_t p = 0; p < instance.num_MPC_parties / 8; p++) {
            //                 __m256i t_u0 = _mm256_loadu_si256((__m256i *)&(inputs_rho_share_v[e][u0][p * 8]));
            //                 __m256i t_u1 = _mm256_loadu_si256((__m256i *)&(inputs_rho_share_v[e][u1][p * 8]));
            //                 t_u1 = _mm256_xor_si256(t_u1, t_u0);
            //                 __m256i rlo_c  = _mm256_xor_si256(t_u1, r_lo);
            //                 __m256i rhi_c  = _mm256_xor_si256(t_u1, r_hi);
            //                 rlo_c = _mm256_i32gather_epi32((const int *)F::mul_lut, rlo_c, 4);
            //                 rhi_c = _mm256_i32gather_epi32((const int *)F::mul_lut, rhi_c, 4);
            //                 t_u1 = _mm256_xor_si256(t_u0, _mm256_xor_si256(rlo_c, rhi_c));
            //                 _mm256_storeu_si256((__m256i *)&(inputs_rho_share_v[e][u][p * 8]), t_u1);
            //             }
            //         }

            //         // AV_v[e][uu + u] = AV_v[e][(uu + u) << 1] ^ (field::GF2E(AV_v[e][((uu + u) << 1) + 1]) * r).data;
            //         __m256i t_av = _mm256_loadu_si256((__m256i *)&(AV_v[e][uu << 1]));
            //         t_av = _mm256_permutevar8x32_epi32(t_av, _mm256_set_epi32(7, 5, 3, 1, 6, 4, 2, 0));
            //         __m128i t_evn = _mm256_castsi256_si128(t_av);
            //         __m128i t_odd = _mm256_extracti128_si256(t_av, 1);
            //         t_av = _mm256_loadu_si256((__m256i *)&(AV_v[e][(uu << 1) + 8]));
            //         t_av = _mm256_permutevar8x32_epi32(t_av, _mm256_set_epi32(7, 5, 3, 1, 6, 4, 2, 0));

            //         __m256i av_evn = _mm256_set_m128i(_mm256_castsi256_si128(t_av), t_evn);
            //         __m256i av_odd = _mm256_set_m128i(_mm256_extracti128_si256(t_av, 1), t_odd);

            //         __m256i rlo_odd = _mm256_xor_si256(av_odd, r_lo);
            //         __m256i rhi_odd = _mm256_xor_si256(av_odd, r_hi);

            //         rlo_odd = _mm256_i32gather_epi32((const int *)F::mul_lut, rlo_odd, 4);
            //         rhi_odd = _mm256_i32gather_epi32((const int *)F::mul_lut, rhi_odd, 4);
            //         t_av = _mm256_xor_si256(av_evn, _mm256_xor_si256(rlo_odd, rhi_odd));
            //         _mm256_storeu_si256((__m256i *)&(AV_v[e][uu]), t_av);

            //         // AG_v[e][uu + u] = AG_v[e][(uu + u) << 1] ^ (field::GF2E(AG_v[e][((uu + u) << 1) + 1]) * r).data;
            //         t_av = _mm256_loadu_si256((__m256i *)&(AG_v[e][uu << 1]));
            //         t_av = _mm256_permutevar8x32_epi32(t_av, _mm256_set_epi32(7, 5, 3, 1, 6, 4, 2, 0));
            //         t_evn = _mm256_castsi256_si128(t_av);
            //         t_odd = _mm256_extracti128_si256(t_av, 1);
            //         t_av = _mm256_loadu_si256((__m256i *)&(AG_v[e][(uu << 1) + 8]));
            //         t_av = _mm256_permutevar8x32_epi32(t_av, _mm256_set_epi32(7, 5, 3, 1, 6, 4, 2, 0));

            //         av_evn = _mm256_set_m128i(_mm256_castsi256_si128(t_av), t_evn);
            //         av_odd = _mm256_set_m128i(_mm256_extracti128_si256(t_av, 1), t_odd);

            //         rlo_odd = _mm256_xor_si256(av_odd, r_lo);
            //         rhi_odd = _mm256_xor_si256(av_odd, r_hi);

            //         rlo_odd = _mm256_i32gather_epi32((const int *)F::mul_lut, rlo_odd, 4);
            //         rhi_odd = _mm256_i32gather_epi32((const int *)F::mul_lut, rhi_odd, 4);
            //         t_av = _mm256_xor_si256(av_evn, _mm256_xor_si256(rlo_odd, rhi_odd));
            //         _mm256_storeu_si256((__m256i *)&(AG_v[e][uu]), t_av);

            //         // AX_v[e][uu + u] = AX_v[e][(uu + u) << 1] ^ (field::GF2E(AX_v[e][((uu + u) << 1) + 1]) * r).data;
            //         t_av = _mm256_loadu_si256((__m256i *)&(AX_v[e][uu << 1]));
            //         t_av = _mm256_permutevar8x32_epi32(t_av, _mm256_set_epi32(7, 5, 3, 1, 6, 4, 2, 0));
            //         t_evn = _mm256_castsi256_si128(t_av);
            //         t_odd = _mm256_extracti128_si256(t_av, 1);
            //         t_av = _mm256_loadu_si256((__m256i *)&(AX_v[e][(uu << 1) + 8]));
            //         t_av = _mm256_permutevar8x32_epi32(t_av, _mm256_set_epi32(7, 5, 3, 1, 6, 4, 2, 0));

            //         av_evn = _mm256_set_m128i(_mm256_castsi256_si128(t_av), t_evn);
            //         av_odd = _mm256_set_m128i(_mm256_extracti128_si256(t_av, 1), t_odd);

            //         rlo_odd = _mm256_xor_si256(av_odd, r_lo);
            //         rhi_odd = _mm256_xor_si256(av_odd, r_hi);

            //         rlo_odd = _mm256_i32gather_epi32((const int *)F::mul_lut, rlo_odd, 4);
            //         rhi_odd = _mm256_i32gather_epi32((const int *)F::mul_lut, rhi_odd, 4);
            //         t_av = _mm256_xor_si256(av_evn, _mm256_xor_si256(rlo_odd, rhi_odd));
            //         _mm256_storeu_si256((__m256i *)&(AX_v[e][uu]), t_av);

            //         // for (int u = 0; u < 8; u++) {
            //         //     // AV_v[e][uu + u] = AV_v[e][(uu + u) << 1] ^ (field::GF2E(AV_v[e][((uu + u) << 1) + 1]) * r).data;
            //         //     // AG_v[e][uu + u] = AG_v[e][(uu + u) << 1] ^ (field::GF2E(AG_v[e][((uu + u) << 1) + 1]) * r).data;
            //         //     // AX_v[e][uu + u] = AX_v[e][(uu + u) << 1] ^ (field::GF2E(AX_v[e][((uu + u) << 1) + 1]) * r).data;
            //         // }
            //     }

#ifdef TIMING
        update_time1 += timing_read(&ctx) - tt;
        tt = timing_read(&ctx);
#endif

            // build table Rho
            mask = 1 << i;
            for (int k = 0; k < mask; k++) {
                Rho[e][k + mask] = field::mul(Rho[e][k], r.data);
                Rho[e][k] ^= Rho[e][k + mask];
            }
#ifdef TIMING
        update_time3 += timing_read(&ctx) - tt;
#endif

            // if (mask >= 8) {
            //     for (int k = 0; k < mask / 8; k++) {
            //         __m256i r_i = _mm256_loadu_si256((__m256i *)&(Rho_v[e][k * 8]));
            //         __m256i rlo_i  = _mm256_xor_si256(r_i, r_lo);
            //         __m256i rhi_i  = _mm256_xor_si256(r_i, r_hi);
            //         rlo_i = _mm256_i32gather_epi32((const int *)F::mul_lut, rlo_i, 4);
            //         rhi_i = _mm256_i32gather_epi32((const int *)F::mul_lut, rhi_i, 4);
            //         __m256i r_j = _mm256_xor_si256(rlo_i, rhi_i);
            //         _mm256_storeu_si256((__m256i *)&(Rho_v[e][k * 8]), _mm256_xor_si256(r_i, r_j));
            //         _mm256_storeu_si256((__m256i *)&(Rho_v[e][k * 8 + mask]), r_j);

            //     }
            // } else
            //     for (int k = 0; k < mask; k++) {
            //         Rho_v[e][k + mask] = (field::GF2E(Rho_v[e][k]) * r).data;
            //         Rho_v[e][k] ^= Rho_v[e][k + mask];
            //     }
        }
    }

    // for (size_t e = 0; e < instance.num_rounds; e++)
    //     for (int i = 0; i < (1 << bit_len); i++)
    //         if (Rho[e][i].data != Rho_v[e][i])
    //             printf("wtf rho\n");

#ifdef TIMING
    tmp_time = timing_read(&ctx);
    printf("gkr phase 1 loop time: %ld\n", tmp_time - start_time);
    printf("gkr phase 1 eval1 time: %ld\n", eval_time1);
    printf("gkr phase 1 eval2 time: %ld\n", eval_time2);
    printf("gkr phase 1 update1 time: %ld\n", update_time1);
    printf("gkr phase 1 update2 time: %ld\n", update_time2);
    printf("gkr phase 1 update3 time: %ld\n", update_time3);
    start_time = timing_read(&ctx);
#endif

    // sumcheck phase 2 init
    for (size_t e = 0; e < instance.num_rounds; e++) {
        Vr[e][0] = AV[e][0];
        auto Vr_sq = field::mul(AV[e][0], AV[e][0]);

        // __m256i vr_lo = _mm256_set1_epi32((uint32_t)(Vr[e][0].data & 255) << 16);
        // __m256i vr_hi = _mm256_set1_epi32((256 + (uint32_t)(Vr[e][0].data >> 8)) << 16);

        AG[e].resize(1 << bit_len);
        AX[e].resize(1 << bit_len);

        for (int i = 0; i < (1 << bit_len); i++) {
            AG[e][i] = 0;
            AX[e][i] = 0;
        }

        for (int i = 0; i < (1 << (bit_len - 1)); i++) {
            int j = (1 << (bit_len - 1)) + i;

            auto tmp = field::mul(Tau[e][i], Rho[e][i]);
            AG[e][j] = field::mul(tmp, AV[e][0]);
            AX[e][j] = field::mul(tmp, Vr_sq);

            tmp = field::mul(Tau[e][j], Rho[e][j]);
            AG[e][i] = field::mul(tmp, AV[e][0]);
            AX[e][i] = field::mul(tmp, Vr_sq);
        }

        // for (int i = 0; i < (1 << (bit_len - 1)); i += 8) {
        //     int j = (1 << (bit_len - 1)) + i;

        //     __m256i ti = _mm256_loadu_si256((__m256i *)&(Tau_v[e][i]));
        //     __m256i tj = _mm256_loadu_si256((__m256i *)&(Tau_v[e][j]));
        //     __m256i ri = _mm256_loadu_si256((__m256i *)&(Rho_v[e][i]));
        //     __m256i rj = _mm256_loadu_si256((__m256i *)&(Rho_v[e][j]));

        //     __m256i ti_lo = _mm256_slli_epi32(_mm256_and_si256(_mm256_set1_epi32(255), ti), 16);
        //     __m256i ti_hi = _mm256_or_si256(
        //             _mm256_slli_epi32(_mm256_and_si256(_mm256_set1_epi32(255 << 8), ti), 8),
        //             _mm256_set1_epi32(256 * 65536));

        //     ti_lo = _mm256_i32gather_epi32((const int *)F::mul_lut, _mm256_xor_si256(ti_lo, ri), 4);
        //     ti_hi = _mm256_i32gather_epi32((const int *)F::mul_lut, _mm256_xor_si256(ti_hi, ri), 4);

        //     __m256i tj_lo = _mm256_slli_epi32(_mm256_and_si256(_mm256_set1_epi32(255), tj), 16);
        //     __m256i tj_hi = _mm256_or_si256(
        //             _mm256_slli_epi32(_mm256_and_si256(_mm256_set1_epi32(255 << 8), tj), 8),
        //             _mm256_set1_epi32(256 * 65536));
        //     tj_lo = _mm256_i32gather_epi32((const int *)F::mul_lut, _mm256_xor_si256(tj_lo, rj), 4);
        //     tj_hi = _mm256_i32gather_epi32((const int *)F::mul_lut, _mm256_xor_si256(tj_hi, rj), 4);

        //     __m256i tmpi = _mm256_xor_si256(tj_lo, tj_hi);
        //     __m256i tmpj = _mm256_xor_si256(ti_lo, ti_hi);

        //     __m256i lo  = _mm256_xor_si256(tmpi, vr_lo);
        //     __m256i hi  = _mm256_xor_si256(tmpi, vr_hi);
        //     lo = _mm256_i32gather_epi32((const int *)F::mul_lut, lo, 4);
        //     hi = _mm256_i32gather_epi32((const int *)F::mul_lut, hi, 4);
        //     tmpi = _mm256_xor_si256(lo, hi);
        //     _mm256_storeu_si256((__m256i *)&(AG_v[e][i]), tmpi);

        //     lo  = _mm256_xor_si256(tmpi, vr_lo);
        //     hi  = _mm256_xor_si256(tmpi, vr_hi);
        //     lo = _mm256_i32gather_epi32((const int *)F::mul_lut, lo, 4);
        //     hi = _mm256_i32gather_epi32((const int *)F::mul_lut, hi, 4);
        //     tmpi = _mm256_xor_si256(lo, hi);
        //     _mm256_storeu_si256((__m256i *)&(AX_v[e][i]), tmpi);

        //     lo  = _mm256_xor_si256(tmpj, vr_lo);
        //     hi  = _mm256_xor_si256(tmpj, vr_hi);
        //     lo = _mm256_i32gather_epi32((const int *)F::mul_lut, lo, 4);
        //     hi = _mm256_i32gather_epi32((const int *)F::mul_lut, hi, 4);
        //     tmpj = _mm256_xor_si256(lo, hi);
        //     _mm256_storeu_si256((__m256i *)&(AG_v[e][j]), tmpj);

        //     lo  = _mm256_xor_si256(tmpj, vr_lo);
        //     hi  = _mm256_xor_si256(tmpj, vr_hi);
        //     lo = _mm256_i32gather_epi32((const int *)F::mul_lut, lo, 4);
        //     hi = _mm256_i32gather_epi32((const int *)F::mul_lut, hi, 4);
        //     tmpj = _mm256_xor_si256(lo, hi);
        //     _mm256_storeu_si256((__m256i *)&(AX_v[e][j]), tmpj);

        //     // for (int k = 0; k < 8; k++) {
        //     //     if (AX_v[e][i + k] != AX[e][i + k].data)
        //     //         printf("wtf ax0\n");
        //     //     if (AX_v[e][j + k] != AX[e][j + k].data)
        //     //         printf("wtf ax1\n");
        //     //     if (AG_v[e][i + k] != AG[e][i + k].data)
        //     //         printf("wtf ag0\n");
        //     //     if (AG_v[e][j + k] != AG[e][j + k].data)
        //     //         printf("wtf ag1\n");
        //     // }
        // }

    }
    // AV = inputs;
    AV = inputs;

#ifdef TIMING
    tmp_time = timing_read(&ctx);
    printf("gkr phase 2 init time: %ld\n", tmp_time - start_time);
    start_time = timing_read(&ctx);
#endif

    // sumcheck phase 2
    for (int i = 0; i < bit_len; i++) {
        // phase 2 evaluate
        coef2[i].resize(instance.num_rounds);
        std::vector<std::vector<field::GF2E>> coef_i_deltas;
        for (size_t e = 0; e < instance.num_rounds; e++) {
            // field::GF2E tmp(0);
            for (int u = 0; u < (1 << (bit_len - i)); u += 2) {
                int one_u = u + 1;
                AV[e][one_u] ^= AV[e][u];
                AG[e][one_u] ^= AG[e][u];
                AX[e][one_u] ^= AX[e][u];

                // tmp += AV[e][one_u] * AX[e][u] + AV[e][u] * AX[e][one_u] + AG[e][one_u];
                // coef2[i][e][0] += field::mul(AV[e][u], AX[e][u]) ^ AG[e][u];
                // coef2[i][e][1] += field::mul(AV[e][one_u], AX[e][one_u]);
                uint16_t tmp0, tmp1;
                field::pair_mul(AV[e][u], AV[e][one_u], AX[e][u], AX[e][one_u], &tmp0, &tmp1);
                coef2[i][e][0] += field::GF2E(tmp0 ^ AG[e][u]);
                coef2[i][e][1] += field::GF2E(tmp1);

            }
            // for (int u = 0; u < (1 << (bit_len - i)); u += 2) {
            //     int one_u = u + 1;
            //     AV_v[e][one_u] ^= AV_v[e][u];
            //     AG_v[e][one_u] ^= AG_v[e][u];
            //     AX_v[e][one_u] ^= AX_v[e][u];

            //     // tmp += AV[e][one_u] * AX[e][u] + AV[e][u] * AX[e][one_u] + AG[e][one_u];
            //     coef2[i][e][0] += field::GF2E(AV_v[e][u]) * field::GF2E(AX_v[e][u]) + field::GF2E(AG_v[e][u]);
            //     coef2[i][e][1] += field::GF2E(AV_v[e][one_u]) * field::GF2E(AX_v[e][one_u]);

            // }
            fr[e][0] -= coef2[i][e][1];
            // if (tmp != fr[e][0])
            //     printf("wtffffffffffff %d\n", i);
        // }

        // for (size_t e = 0; e < instance.num_rounds; e++) {
            std::vector<field::GF2E> coef_ie_deltas(2);
            __m256i coef_ie_deltas_v[3];
            for (int k = 0; k < 3; k++)
                coef_ie_deltas_v[k] = _mm256_setzero_si256();

            for (size_t k = 0; k < instance.num_MPC_parties / 16; k++) {
                for (size_t p = 0; p < 16; p++) {
                    auto random_coef_share =
                        random_tapes.get_bytes(e, k * 16 + p,
                                instance.aes_params.key_size +
                                instance.aes_params.num_sboxes +
                                3 * bit_len * instance.lambda +
                                2 * i * instance.lambda,
                                2 * instance.lambda);
                    coef_shares_v[e][i][3][k * 32 + p] = *(uint8_t *)(random_coef_share.data());
                    coef_shares_v[e][i][4][k * 32 + p] = *(uint8_t *)(random_coef_share.data() + 2);
                    coef_shares_v[e][i][3][k * 32 + 16 + p] = *(uint8_t *)(random_coef_share.data() + 1);
                    coef_shares_v[e][i][4][k * 32 + 16 + p] = *(uint8_t *)(random_coef_share.data() + 3);
                }
                __m256i t_coef0 = _mm256_loadu_si256((__m256i *)&(coef_shares_v[e][i][3][k * 32]));
                __m256i t_coef1 = _mm256_loadu_si256((__m256i *)&(coef_shares_v[e][i][4][k * 32]));
                __m256i t_fr    = _mm256_loadu_si256((__m256i *)&(fr_shares_v[e][k * 32]));
                t_fr = _mm256_xor_si256(t_coef1, t_fr);
                _mm256_storeu_si256((__m256i *)&(fr_shares_v[e][k * 32]), t_fr);
                coef_ie_deltas_v[0] = _mm256_xor_si256(coef_ie_deltas_v[0], t_coef0);
                coef_ie_deltas_v[1] = _mm256_xor_si256(coef_ie_deltas_v[1], t_coef1);
            }
            coef_ie_deltas[0] = coef2[i][e][0] - field::GF2E(_mm256_hxor_epu16(coef_ie_deltas_v[0]));
            coef_ie_deltas[1] = coef2[i][e][1] - field::GF2E(_mm256_hxor_epu16(coef_ie_deltas_v[1]));
            coef_i_deltas.push_back(coef_ie_deltas);

            coef_shares_v[e][i][3][0] ^= (coef_ie_deltas[0].data & 255);
            coef_shares_v[e][i][4][0] ^= (coef_ie_deltas[1].data & 255);
            coef_shares_v[e][i][3][16] ^= (coef_ie_deltas[0].data >> 8);
            coef_shares_v[e][i][4][16] ^= (coef_ie_deltas[1].data >> 8);

            fr_shares_v[e][0] ^= coef_ie_deltas[1].data & 255;
            fr_shares_v[e][16] ^= coef_ie_deltas[1].data >> 8;


            // for (size_t p = 0; p < instance.num_MPC_parties; p++) {
            //     auto random_coef_share =
            //         random_tapes.get_bytes(e, p,
            //                 instance.aes_params.key_size +
            //                 instance.aes_params.num_sboxes +
            //                 3 * bit_len * instance.lambda +
            //                 2 * i * instance.lambda,
            //                 2 * instance.lambda);
            //     coef_shares[e][i][3][p].from_bytes(random_coef_share.data());
            //     coef_shares[e][i][4][p].from_bytes(random_coef_share.data() + instance.lambda);

            //     fr_shares[e][p] -= coef_shares[e][i][4][p];
            //     coef_ie_deltas[0] += coef_shares[e][i][3][p];
            //     coef_ie_deltas[1] += coef_shares[e][i][4][p];
            // }
            // coef_ie_deltas[0] = coef2[i][e][0] - coef_ie_deltas[0];
            // coef_ie_deltas[1] = coef2[i][e][1] - coef_ie_deltas[1];
            // coef_i_deltas.push_back(coef_ie_deltas);
            // coef_shares[e][i][3][0] += coef_ie_deltas[0];
            // coef_shares[e][i][4][0] += coef_ie_deltas[1];
            // fr_shares[e][0] += coef_ie_deltas[1];
        }
        coef2_deltas.push_back(coef_i_deltas);

        h_i =
            // phase_gkr_commitment(instance, salt, keypair.second, message, message_len,
            //                    party_seed_commitments, rep_key_deltas, rep_t_deltas,
            //                    rep_output_broadcasts);
            phase_sumcheck_commitment(instance, salt, h_i, coef2_deltas[i]);

        std::vector<field::GF2E> sgm = phase_sumcheck_expand(instance, h_i);
        for (size_t e = 0; e < instance.num_rounds; e++) {
            auto r = sgm[e];
            auto r_raw = r.data;
            // __m256i r_lo = _mm256_set1_epi32((uint32_t)(r.data & 255) << 16);
            // __m256i r_hi = _mm256_set1_epi32((256 + (uint32_t)(r.data >> 8)) << 16);
            fr[e][0] = (coef2[i][e][1] * r + fr[e][0]) * r + coef2[i][e][0];
            // for (size_t p = 0; p < instance.num_MPC_parties; p++) {
            //     fr_shares[e][p] = (coef_shares[e][i][4][p] * r + fr_shares[e][p]) * r + coef_shares[e][i][3][p];
            // }
            __m256i table0 = _mm256_loadu_si256((__m256i *)&(field::GF2E::mul_lut[r_raw][0]));
            __m256i table1 = _mm256_loadu_si256((__m256i *)&(field::GF2E::mul_lut[r_raw][8]));
            __m256i table2 = _mm256_loadu_si256((__m256i *)&(field::GF2E::mul_lut[r_raw][16]));
            __m256i table3 = _mm256_loadu_si256((__m256i *)&(field::GF2E::mul_lut[r_raw][24]));
            __m256i mask1 = _mm256_set1_epi8(0x0f);
            __m256i mask2 = _mm256_set1_epi8(0xf0);
            for (size_t k = 0; k < instance.num_MPC_parties / 16; k++) {
                __m256i lo, hi, tmp;
                __m256i t_coef0 = _mm256_loadu_si256((__m256i *)&(coef_shares_v[e][i][3][k * 32]));
                __m256i t_coef1 = _mm256_loadu_si256((__m256i *)&(coef_shares_v[e][i][4][k * 32]));
                __m256i t_fr    = _mm256_loadu_si256((__m256i *)&(fr_shares_v[e][k * 32]));

                // coef1 * r
                lo = _mm256_and_si256(t_coef1, mask1);  // a0, a2
                lo = _mm256_shuffle_epi8(table0, lo);   // a0lo, a2hi
                hi = _mm256_srli_epi64(_mm256_and_si256(t_coef1, mask2), 4);    // a1, a3
                hi = _mm256_shuffle_epi8(table1, hi);   // a1lo, a3hi
                tmp = _mm256_xor_si256(hi, lo);
                t_coef1 = _mm256_permute4x64_epi64(t_coef1, 2 | (3 << 2) | (1 << 6));   // a2,a3; a0,a1
                lo = _mm256_and_si256(t_coef1, mask1);  // a2, a0
                lo = _mm256_shuffle_epi8(table2, lo);   // a2lo, a0hi
                hi = _mm256_srli_epi64(_mm256_and_si256(t_coef1, mask2), 4);    // a3, a1
                hi = _mm256_shuffle_epi8(table3, hi);   // a3lo, a1hi
                t_coef1 = _mm256_xor_si256(tmp, _mm256_xor_si256(hi, lo));
                // + fr
                t_coef1 = _mm256_xor_si256(t_fr, t_coef1);
                // * r
                lo = _mm256_and_si256(t_coef1, mask1);  // a0, a2
                lo = _mm256_shuffle_epi8(table0, lo);   // a0lo, a2hi
                hi = _mm256_srli_epi64(_mm256_and_si256(t_coef1, mask2), 4);    // a1, a3
                hi = _mm256_shuffle_epi8(table1, hi);   // a1lo, a3hi
                tmp = _mm256_xor_si256(hi, lo);
                t_coef1 = _mm256_permute4x64_epi64(t_coef1, 2 | (3 << 2) | (1 << 6));   // a2,a3; a0,a1
                lo = _mm256_and_si256(t_coef1, mask1);  // a2, a0
                lo = _mm256_shuffle_epi8(table2, lo);   // a2lo, a0hi
                hi = _mm256_srli_epi64(_mm256_and_si256(t_coef1, mask2), 4);    // a3, a1
                hi = _mm256_shuffle_epi8(table3, hi);   // a3lo, a1hi
                t_coef1 = _mm256_xor_si256(tmp, _mm256_xor_si256(hi, lo));
                // + coef[0]
                t_coef1 = _mm256_xor_si256(t_coef0, t_coef1);

                _mm256_storeu_si256((__m256i *)&(fr_shares_v[e][k * 32]), t_coef1);

                // for (int p = 0; p < 16; p++) {
                //     if (fr_shares_v[e][k * 32 + p] != (fr_shares[e][k * 16 + p].data & 255))
                //         printf("wtf0\n");
                //     if (fr_shares_v[e][k * 32 + p + 16] != (fr_shares[e][k * 16 + p].data >> 8))
                //         printf("wtf1\n");
                // }
            }


            int mask = 1 << (bit_len - i - 1);
            // if (mask < 65536)
                for (int u = 0; u < mask; u++) {
                    int u0 = u << 1;
                    int u1 = u0 + 1;
                    // for (size_t p = 0; p < instance.num_MPC_parties; p++)
                    //     inputs_sgm_share[e][u][p] = inputs_sgm_share[e][u0][p] + (inputs_sgm_share[e][u1][p] - inputs_sgm_share[e][u0][p]) * r;
                    for (size_t k = 0; k < instance.num_MPC_parties / 16; k++) {
                        __m256i lo, hi, tmp;
                        __m256i t_u0 = _mm256_loadu_si256((__m256i *)&(inputs_sgm_share_v[e][u0][k * 32]));
                        __m256i t_u1 = _mm256_loadu_si256((__m256i *)&(inputs_sgm_share_v[e][u1][k * 32]));
                        t_u1 = _mm256_xor_si256(t_u0, t_u1);

                        // (t_u1 + t_u0) * r
                        lo = _mm256_and_si256(t_u1, mask1);  // a0, a2
                        lo = _mm256_shuffle_epi8(table0, lo);   // a0lo, a2hi
                        hi = _mm256_srli_epi64(_mm256_and_si256(t_u1, mask2), 4);    // a1, a3
                        hi = _mm256_shuffle_epi8(table1, hi);   // a1lo, a3hi
                        tmp = _mm256_xor_si256(hi, lo);
                        t_u1 = _mm256_permute4x64_epi64(t_u1, 2 | (3 << 2) | (1 << 6));   // a2,a3; a0,a1
                        lo = _mm256_and_si256(t_u1, mask1);  // a2, a0
                        lo = _mm256_shuffle_epi8(table2, lo);   // a2lo, a0hi
                        hi = _mm256_srli_epi64(_mm256_and_si256(t_u1, mask2), 4);    // a3, a1
                        hi = _mm256_shuffle_epi8(table3, hi);   // a3lo, a1hi
                        t_u1 = _mm256_xor_si256(tmp, _mm256_xor_si256(hi, lo));

                        // + t_u0
                        t_u1 = _mm256_xor_si256(t_u0, t_u1);
                        _mm256_storeu_si256((__m256i *)&(inputs_sgm_share_v[e][u][k * 32]), t_u1);
                    }

                    // field::pair_mul(AV[e][u1], AG[e][u1], r.data, r.data, &AV[e][u1], &AG[e][u1]);
                    // AV[e][u] = AV[e][u0] ^ AV[e][u1];
                    // AG[e][u] = AG[e][u0] ^ AG[e][u1];
                    // AX[e][u] = AX[e][u0] ^ field::mul(AX[e][u1], r.data);
                    field::scale_mul(AV[e][u1], AG[e][u1], AX[e][u1], r.data,
                            AV[e][u0], AG[e][u0], AX[e][u0],
                            &AV[e][u], &AG[e][u], &AX[e][u]);
                    // AV[e][u] = AV[e][u0] ^ field::mul(AV[e][u1], r.data);
                    // AG[e][u] = AG[e][u0] ^ field::mul(AG[e][u1], r.data);
                    // AX[e][u] = AX[e][u0] ^ field::mul(AX[e][u1], r.data);
                }
            // else
            //     for (int uu = 0; uu < mask / 16; uu++) {
            //         for (int u = 0; u < 16; u++) {
            //             int u0 = (uu * 16 + u) << 1;
            //             int u1 = u0 + 1;
            //             // for (size_t p = 0; p < instance.num_MPC_parties; p++)
            //             //     inputs_sgm_share[e][u][p] = inputs_sgm_share[e][u0][p] + (inputs_sgm_share[e][u1][p] - inputs_sgm_share[e][u0][p]) * r;
            //             for (size_t k = 0; k < instance.num_MPC_parties / 16; k++) {
            //                 __m256i lo, hi, tmp;
            //                 __m256i t_u0 = _mm256_loadu_si256((__m256i *)&(inputs_sgm_share_v[e][u0][k * 32]));
            //                 __m256i t_u1 = _mm256_loadu_si256((__m256i *)&(inputs_sgm_share_v[e][u1][k * 32]));
            //                 t_u1 = _mm256_xor_si256(t_u0, t_u1);

            //                 // (t_u1 + t_u0) * r
            //                 lo = _mm256_and_si256(t_u1, mask1);  // a0, a2
            //                 lo = _mm256_shuffle_epi8(table0, lo);   // a0lo, a2hi
            //                 hi = _mm256_srli_epi64(_mm256_and_si256(t_u1, mask2), 4);    // a1, a3
            //                 hi = _mm256_shuffle_epi8(table1, hi);   // a1lo, a3hi
            //                 tmp = _mm256_xor_si256(hi, lo);
            //                 t_u1 = _mm256_permute4x64_epi64(t_u1, 2 | (3 << 2) | (1 << 6));   // a2,a3; a0,a1
            //                 lo = _mm256_and_si256(t_u1, mask1);  // a2, a0
            //                 lo = _mm256_shuffle_epi8(table2, lo);   // a2lo, a0hi
            //                 hi = _mm256_srli_epi64(_mm256_and_si256(t_u1, mask2), 4);    // a3, a1
            //                 hi = _mm256_shuffle_epi8(table3, hi);   // a3lo, a1hi
            //                 t_u1 = _mm256_xor_si256(tmp, _mm256_xor_si256(hi, lo));

            //                 // + t_u0
            //                 t_u1 = _mm256_xor_si256(t_u0, t_u1);
            //                 _mm256_storeu_si256((__m256i *)&(inputs_sgm_share_v[e][uu * 16 + u][k * 32]), t_u1);
            //             }
            //         }

            //         __m256i lo, hi, tmp;
            //         __m256i s_u0, s_u1, v_u0, v_u1;
            //         // AV
            //         s_u0 = _mm256_loadu_si256((__m256i *)&(AV[e][uu * 16 * 2]));
            //         s_u1 = _mm256_loadu_si256((__m256i *)&(AV[e][uu * 16 * 2 + 16]));
            //         v_u0 = _mm256_packus_epi32(
            //                 _mm256_and_si256(s_u0, _mm256_set1_epi32(0xffff)),
            //                 _mm256_and_si256(s_u1, _mm256_set1_epi32(0xffff)));
            //         v_u0 = _mm256_permute4x64_epi64(v_u0, 0b11011000);
            //         v_u1 = _mm256_packus_epi32(
            //                 _mm256_srli_epi32(s_u0, 16),
            //                 _mm256_srli_epi32(s_u1, 16));
            //         v_u1 = _mm256_permute4x64_epi64(v_u1, 0b11011000);
            //         // to alt mapping
            //         lo = _mm256_packus_epi16(_mm256_and_si256(v_u1, _mm256_set1_epi16(0xff)), _mm256_set1_epi64x(0));
            //         lo = _mm256_permute4x64_epi64(lo, 0b11011000);
            //         hi = _mm256_packus_epi16(_mm256_set1_epi64x(0), _mm256_srli_epi16(v_u1, 8));
            //         hi = _mm256_permute4x64_epi64(hi, 0b11011000);
            //         v_u1 = _mm256_xor_si256(lo, hi);

            //         // u1 * r
            //         lo = _mm256_and_si256(v_u1, mask1);  // a0, a2
            //         lo = _mm256_shuffle_epi8(table0, lo);   // a0lo, a2hi
            //         hi = _mm256_srli_epi64(_mm256_and_si256(v_u1, mask2), 4);    // a1, a3
            //         hi = _mm256_shuffle_epi8(table1, hi);   // a1lo, a3hi
            //         tmp = _mm256_xor_si256(hi, lo);
            //         v_u1 = _mm256_permute4x64_epi64(v_u1, 2 | (3 << 2) | (1 << 6));   // a2,a3; a0,a1
            //         lo = _mm256_and_si256(v_u1, mask1);  // a2, a0
            //         lo = _mm256_shuffle_epi8(table2, lo);   // a2lo, a0hi
            //         hi = _mm256_srli_epi64(_mm256_and_si256(v_u1, mask2), 4);    // a3, a1
            //         hi = _mm256_shuffle_epi8(table3, hi);   // a3lo, a1hi
            //         v_u1 = _mm256_xor_si256(tmp, _mm256_xor_si256(hi, lo));

            //         // to standard mapping
            //         v_u1 = _mm256_unpacklo_epi8(
            //                 _mm256_permute4x64_epi64(v_u1, 0b11011000),
            //                 _mm256_permute4x64_epi64(v_u1, 0b01110010));

            //         v_u1 = _mm256_xor_si256(v_u0, v_u1);

            //         _mm256_storeu_si256((__m256i *)&(AV[e][uu * 16]), v_u1);

            //         // AG
            //         s_u0 = _mm256_loadu_si256((__m256i *)&(AG[e][uu * 16 * 2]));
            //         s_u1 = _mm256_loadu_si256((__m256i *)&(AG[e][uu * 16 * 2 + 16]));
            //         v_u0 = _mm256_packus_epi32(
            //                 _mm256_and_si256(s_u0, _mm256_set1_epi32(0xffff)),
            //                 _mm256_and_si256(s_u1, _mm256_set1_epi32(0xffff)));
            //         v_u0 = _mm256_permute4x64_epi64(v_u0, 0b11011000);
            //         v_u1 = _mm256_packus_epi32(
            //                 _mm256_srli_epi32(s_u0, 16),
            //                 _mm256_srli_epi32(s_u1, 16));
            //         v_u1 = _mm256_permute4x64_epi64(v_u1, 0b11011000);
            //         // to alt mapping
            //         lo = _mm256_packus_epi16(_mm256_and_si256(v_u1, _mm256_set1_epi16(0xff)), _mm256_set1_epi64x(0));
            //         lo = _mm256_permute4x64_epi64(lo, 0b11011000);
            //         hi = _mm256_packus_epi16(_mm256_set1_epi64x(0), _mm256_srli_epi16(v_u1, 8));
            //         hi = _mm256_permute4x64_epi64(hi, 0b11011000);
            //         v_u1 = _mm256_xor_si256(lo, hi);

            //         // u1 * r
            //         lo = _mm256_and_si256(v_u1, mask1);  // a0, a2
            //         lo = _mm256_shuffle_epi8(table0, lo);   // a0lo, a2hi
            //         hi = _mm256_srli_epi64(_mm256_and_si256(v_u1, mask2), 4);    // a1, a3
            //         hi = _mm256_shuffle_epi8(table1, hi);   // a1lo, a3hi
            //         tmp = _mm256_xor_si256(hi, lo);
            //         v_u1 = _mm256_permute4x64_epi64(v_u1, 2 | (3 << 2) | (1 << 6));   // a2,a3; a0,a1
            //         lo = _mm256_and_si256(v_u1, mask1);  // a2, a0
            //         lo = _mm256_shuffle_epi8(table2, lo);   // a2lo, a0hi
            //         hi = _mm256_srli_epi64(_mm256_and_si256(v_u1, mask2), 4);    // a3, a1
            //         hi = _mm256_shuffle_epi8(table3, hi);   // a3lo, a1hi
            //         v_u1 = _mm256_xor_si256(tmp, _mm256_xor_si256(hi, lo));

            //         // to standard mapping
            //         v_u1 = _mm256_unpacklo_epi8(
            //                 _mm256_permute4x64_epi64(v_u1, 0b11011000),
            //                 _mm256_permute4x64_epi64(v_u1, 0b01110010));

            //         v_u1 = _mm256_xor_si256(v_u0, v_u1);

            //         _mm256_storeu_si256((__m256i *)&(AG[e][uu * 16]), v_u1);

            //         // AX
            //         s_u0 = _mm256_loadu_si256((__m256i *)&(AX[e][uu * 16 * 2]));
            //         s_u1 = _mm256_loadu_si256((__m256i *)&(AX[e][uu * 16 * 2 + 16]));
            //         v_u0 = _mm256_packus_epi32(
            //                 _mm256_and_si256(s_u0, _mm256_set1_epi32(0xffff)),
            //                 _mm256_and_si256(s_u1, _mm256_set1_epi32(0xffff)));
            //         v_u0 = _mm256_permute4x64_epi64(v_u0, 0b11011000);
            //         v_u1 = _mm256_packus_epi32(
            //                 _mm256_srli_epi32(s_u0, 16),
            //                 _mm256_srli_epi32(s_u1, 16));
            //         v_u1 = _mm256_permute4x64_epi64(v_u1, 0b11011000);
            //         // to alt mapping
            //         lo = _mm256_packus_epi16(_mm256_and_si256(v_u1, _mm256_set1_epi16(0xff)), _mm256_set1_epi64x(0));
            //         lo = _mm256_permute4x64_epi64(lo, 0b11011000);
            //         hi = _mm256_packus_epi16(_mm256_set1_epi64x(0), _mm256_srli_epi16(v_u1, 8));
            //         hi = _mm256_permute4x64_epi64(hi, 0b11011000);
            //         v_u1 = _mm256_xor_si256(lo, hi);

            //         // u1 * r
            //         lo = _mm256_and_si256(v_u1, mask1);  // a0, a2
            //         lo = _mm256_shuffle_epi8(table0, lo);   // a0lo, a2hi
            //         hi = _mm256_srli_epi64(_mm256_and_si256(v_u1, mask2), 4);    // a1, a3
            //         hi = _mm256_shuffle_epi8(table1, hi);   // a1lo, a3hi
            //         tmp = _mm256_xor_si256(hi, lo);
            //         v_u1 = _mm256_permute4x64_epi64(v_u1, 2 | (3 << 2) | (1 << 6));   // a2,a3; a0,a1
            //         lo = _mm256_and_si256(v_u1, mask1);  // a2, a0
            //         lo = _mm256_shuffle_epi8(table2, lo);   // a2lo, a0hi
            //         hi = _mm256_srli_epi64(_mm256_and_si256(v_u1, mask2), 4);    // a3, a1
            //         hi = _mm256_shuffle_epi8(table3, hi);   // a3lo, a1hi
            //         v_u1 = _mm256_xor_si256(tmp, _mm256_xor_si256(hi, lo));

            //         // to standard mapping
            //         v_u1 = _mm256_unpacklo_epi8(
            //                 _mm256_permute4x64_epi64(v_u1, 0b11011000),
            //                 _mm256_permute4x64_epi64(v_u1, 0b01110010));

            //         v_u1 = _mm256_xor_si256(v_u0, v_u1);

            //         _mm256_storeu_si256((__m256i *)&(AX[e][uu * 16]), v_u1);

            //         // for (int u = 0; u < 16; u++) {
            //         //     int u0 = (uu * 16 + u) << 1;
            //         //     int u1 = u0 + 1;
            //         //     AV[e][uu * 16 + u] = AV[e][u0] ^ field::mul(AV[e][u1], r.data);
            //         //     AG[e][uu * 16 + u] = AG[e][u0] ^ field::mul(AG[e][u1], r.data);
            //         //     AX[e][uu * 16 + u] = AX[e][u0] ^ field::mul(AX[e][u1], r.data);
            //         // }
            //     }

            // update pred
            field::GF2E one(1);
            if (i < (bit_len - 1))
                pred[e] *= tau[i][e] * rho[i][e] + (tau[i][e] + rho[i][e] + one) * (r + one);
            else
                pred[e] *= tau[i][e] * rho[i][e] + (tau[i][e] + rho[i][e] + one) * r;
        }
    }

#ifdef TIMING
    tmp_time = timing_read(&ctx);
    printf("gkr phase 2 loop time: %ld\n", tmp_time - start_time);
    start_time = timing_read(&ctx);
#endif

    for (size_t e = 0; e < instance.num_rounds; e++) {
        Vs[e][0] = AV[e][0];
        for (size_t k = 0; k < instance.num_MPC_parties / 16; k++)
            for (size_t p = 0; p < 16; p++) {
                auto Vr = Vr_share.get(e, k * 16 + p);
                auto Vs = Vs_share.get(e, k * 16 + p);
                auto fr = fr_share.get(e, k * 16 + p);
                fr[0] = field::GF2E(fr_shares_v[e][k * 32 + p] | ((uint16_t)fr_shares_v[e][k * 32 + p + 16] << 8));
                Vr[0] = field::GF2E(inputs_rho_share_v[e][0][k * 32 + p] | ((uint16_t)inputs_rho_share_v[e][0][k * 32 + p + 16] << 8));
                Vs[0] = field::GF2E(inputs_sgm_share_v[e][0][k * 32 + p] | ((uint16_t)inputs_sgm_share_v[e][0][k * 32 + p + 16] << 8));
                // Vr[0] = inputs_rho_share[e][0][k * 16 + p];
                // Vs[0] = inputs_sgm_share[e][0][k * 16 + p];
            }
    }

#ifdef TIMING
    tmp_time = timing_read(&ctx);
    printf("gkr final time: %ld\n", tmp_time - start_time);
    start_time = timing_read(&ctx);
#endif

    // next phase: prove that
    // sum(broadcast[0]) = pred * sum(broadcast[1])(1 - sum(broadcast[1]) * sum(broadcast[2]))
    // a + pred * b * (1 + b * c) = 0

    // do a sanity check here
    // for (size_t e = 0; e < instance.num_rounds; e++) {
    //     field::GF2E tmp0(0);
    //     field::GF2E tmp1(0);
    //     field::GF2E tmp2(0);
    //     for (size_t p = 0; p < instance.num_MPC_parties; p++) {
    //         auto a = fr_share.get(e, p);
    //         auto b = Vr_share.get(e, p);
    //         auto c = Vs_share.get(e, p);
    //         tmp0 += a[0];
    //         tmp1 += b[0];
    //         tmp2 += c[0];
    //     }
    //     if (tmp1 != Vr[e][0])
    //         throw std::runtime_error("sanity check Vr failed");
    //     if (tmp2 != Vs[e][0])
    //         throw std::runtime_error("sanity check Vs failed");
    //     if (tmp0 != fr[e][0])
    //         throw std::runtime_error("sanity check fr failed");
    //     if (tmp0 != pred[e] * tmp1 * (field::GF2E(1) - tmp1 * tmp2))
    //         throw std::runtime_error("sanity check GKR failed");
    // }

    // expand challenge hash to M * m1 values
    // std::vector<std::vector<field::GF2E>> r_ejs = phase_1_expand(instance, h_1);

    /////////////////////////////////////////////////////////////////////////////
    // phase 3: commit to the checking polynomials
    /////////////////////////////////////////////////////////////////////////////

    // a vector of the first m2+1 field elements for interpolation
    // m2 = 1
    std::vector<field::GF2E> x_values_for_interpolation_zero_to_m2 = field::get_first_n_field_elements(1 + 1);
    std::vector<std::vector<field::GF2E>> precomputation_for_zero_to_m2 = precompute_lagrange_polynomials(x_values_for_interpolation_zero_to_m2);

    std::vector<field::GF2E> x_values_for_interpolation_zero_to_3m2 = field::get_first_n_field_elements(3 * 1 + 1);
    std::vector<std::vector<field::GF2E>> precomputation_for_zero_to_3m2 = precompute_lagrange_polynomials(x_values_for_interpolation_zero_to_3m2);

    std::vector<std::vector<std::vector<field::GF2E>>> fr_poly_share(instance.num_rounds);
    std::vector<std::vector<std::vector<field::GF2E>>> Vr_poly_share(instance.num_rounds);
    std::vector<std::vector<std::vector<field::GF2E>>> Vs_poly_share(instance.num_rounds);
    std::vector<std::vector<std::vector<field::GF2E>>> P_share(instance.num_rounds);
    std::vector<std::vector<field::GF2E>> fr_poly(instance.num_rounds);
    std::vector<std::vector<field::GF2E>> Vr_poly(instance.num_rounds);
    std::vector<std::vector<field::GF2E>> Vs_poly(instance.num_rounds);
    std::vector<std::vector<field::GF2E>> P_poly(instance.num_rounds);

    std::vector<std::vector<field::GF2E>> P(instance.num_rounds);
    std::vector<std::vector<field::GF2E>> P_deltas(instance.num_rounds);
    std::vector<field::GF2E> one(1, field::GF2E(1));

    for (size_t e = 0; e < instance.num_rounds; e++) {
        fr_poly_share[e].resize(instance.num_MPC_parties);
        Vr_poly_share[e].resize(instance.num_MPC_parties);
        Vs_poly_share[e].resize(instance.num_MPC_parties);
        P_share[e].resize(instance.num_MPC_parties);
        P[e].resize(2 * 1 + 1);
        P_deltas[e].resize(2 * 1 + 1);

        for (size_t p = 0; p < instance.num_MPC_parties; p++) {
            auto a = fr_share.get(e, p);
            auto b = Vr_share.get(e, p);
            auto c = Vs_share.get(e, p);

            auto random_share = random_tapes.get_bytes(e, p,
                        instance.aes_params.key_size +
                        instance.aes_params.num_sboxes +
                        3 * bit_len * instance.lambda +
                        2 * bit_len * instance.lambda,
                        3 * instance.lambda + 3 * instance.lambda);

            a[1].from_bytes(random_share.data());
            b[1].from_bytes(random_share.data() + instance.lambda);
            c[1].from_bytes(random_share.data() + 2 * instance.lambda);
            fr[e][1] += a[1];
            Vr[e][1] += b[1];
            Vs[e][1] += c[1];
            fr_poly_share[e][p] = interpolate_with_precomputation(precomputation_for_zero_to_m2, a);
            Vr_poly_share[e][p] = interpolate_with_precomputation(precomputation_for_zero_to_m2, b);
            Vs_poly_share[e][p] = interpolate_with_precomputation(precomputation_for_zero_to_m2, c);

            P_share[e][p].resize(3 * 1 + 1);
            P_share[e][p][1].from_bytes(random_share.data() + 3 * instance.lambda);
            P_share[e][p][2].from_bytes(random_share.data() + 4 * instance.lambda);
            P_share[e][p][3].from_bytes(random_share.data() + 5 * instance.lambda);
            P_deltas[e][0] += P_share[e][p][1];
            P_deltas[e][1] += P_share[e][p][2];
            P_deltas[e][2] += P_share[e][p][3];
        }
        fr_poly[e] = interpolate_with_precomputation(precomputation_for_zero_to_m2, fr[e]);
        Vr_poly[e] = interpolate_with_precomputation(precomputation_for_zero_to_m2, Vr[e]);
        Vs_poly[e] = interpolate_with_precomputation(precomputation_for_zero_to_m2, Vs[e]);
        auto P = fr_poly[e] + pred[e] * Vr_poly[e] * (one + Vr_poly[e] * Vs_poly[e]);

        for (size_t k = 1; k <= 3 * 1; k++) {
            // calculate offset
            field::GF2E k_element = x_values_for_interpolation_zero_to_3m2[k];
            P_deltas[e][k - 1] = eval(P, k_element) - P_deltas[e][k - 1];
            // adjust first share
            P_share[e][0][k] += P_deltas[e][k - 1];
        }
    }

#ifdef TIMING
    tmp_time = timing_read(&ctx);
    printf("linear pcp time: %ld\n", tmp_time - start_time);
#endif

    /////////////////////////////////////////////////////////////////////////////
    // phase 4: challenge the checking polynomials
    /////////////////////////////////////////////////////////////////////////////

    std::vector<uint8_t> h_2 = phase_2_commitment(instance, salt, h_i, P_deltas);

    // expand challenge hash to M values

    std::vector<field::GF2E> forbidden_challenge_values = field::get_first_n_field_elements(1);
    std::vector<field::GF2E> R_es = phase_2_expand(instance, h_2, forbidden_challenge_values);

    /////////////////////////////////////////////////////////////////////////////
    // phase 5: commit to the views of the checking protocol
    /////////////////////////////////////////////////////////////////////////////

    // std::vector<field::GF2E> d(instance.num_rounds);
    std::vector<std::vector<field::GF2E>> d_shares(instance.num_rounds);

    std::vector<field::GF2E> a(instance.num_rounds);
    std::vector<field::GF2E> b(instance.num_rounds);
    std::vector<field::GF2E> c(instance.num_rounds);
    std::vector<std::vector<field::GF2E>> a_shares(instance.num_rounds);
    std::vector<std::vector<field::GF2E>> b_shares(instance.num_rounds);
    std::vector<std::vector<field::GF2E>> c_shares(instance.num_rounds);

    std::vector<field::GF2E> lagrange_polys_evaluated_at_Re_m2(1 + 1);
    std::vector<field::GF2E> lagrange_polys_evaluated_at_Re_3m2(3 * 1 + 1);

    for (size_t e = 0; e < instance.num_rounds; e++) {
        a_shares[e].resize(instance.num_MPC_parties);
        b_shares[e].resize(instance.num_MPC_parties);
        c_shares[e].resize(instance.num_MPC_parties);
        d_shares[e].resize(instance.num_MPC_parties);
        for (size_t k = 0; k < 1 + 1; k++)
            lagrange_polys_evaluated_at_Re_m2[k] = eval(precomputation_for_zero_to_m2[k], R_es[e]);
        for (size_t k = 0; k < 3 * 1 + 1; k++)
            lagrange_polys_evaluated_at_Re_3m2[k] = eval(precomputation_for_zero_to_3m2[k], R_es[e]);

        for (size_t p = 0; p < instance.num_MPC_parties; p++) {
            auto fr   = fr_share.get(e, p);
            auto Vr = Vr_share.get(e, p);
            auto Vs = Vs_share.get(e, p);
            a_shares[e][p] = dot_product(lagrange_polys_evaluated_at_Re_m2, fr);
            b_shares[e][p] = dot_product(lagrange_polys_evaluated_at_Re_m2, Vr);
            c_shares[e][p] = dot_product(lagrange_polys_evaluated_at_Re_m2, Vs);

          // compute c_e^i
            d_shares[e][p] = dot_product(lagrange_polys_evaluated_at_Re_3m2, P_share[e][p]);

        }

        // open d_e and a,b,c values
        for (size_t p = 0; p < instance.num_MPC_parties; p++) {
            a[e] += a_shares[e][p];
            b[e] += b_shares[e][p];
            c[e] += c_shares[e][p];
        }
    }

    /////////////////////////////////////////////////////////////////////////////
    // phase 6: challenge the views of the checking protocol
    /////////////////////////////////////////////////////////////////////////////

    std::vector<uint8_t> h_3 = phase_3_commitment(
            instance, salt, h_2, d_shares, a, a_shares, b, b_shares, c, c_shares);

    std::vector<uint16_t> missing_parties = phase_3_expand(instance, h_3);

    /////////////////////////////////////////////////////////////////////////////
    // phase 7: Open the views of the checking protocol
    /////////////////////////////////////////////////////////////////////////////
    std::vector<reveal_list_t> seeds;
    for (size_t e = 0; e < instance.num_rounds; e++) {
        seeds.push_back(
                seed_trees[e].reveal_all_but(missing_parties[e]));
    }
    // build signature
    std::vector<dubhe_repetition_proof_t> proofs;
    for (size_t e = 0; e < instance.num_rounds; e++) {
        size_t missing_party = missing_parties[e];
        std::vector<uint8_t> commitment(instance.digest_size);
        auto missing_commitment =
            party_seed_commitments.get(e, missing_party);
        std::copy(std::begin(missing_commitment), std::end(missing_commitment),
                std::begin(commitment));
        std::vector<std::vector<field::GF2E>> coef1(bit_len);
        std::vector<std::vector<field::GF2E>> coef2(bit_len);
        for (int i = 0; i < bit_len; i++) {
            coef1[i] = coef1_deltas[i][e];
            coef2[i] = coef2_deltas[i][e];
        }
        dubhe_repetition_proof_t proof{
            seeds[e],
            commitment,
            rep_key_deltas[e],
            rep_t_deltas[e],
            coef1,
            coef2,
            P_deltas[e],
            a[e],
            b[e],
            c[e],
        };

        // sanity check c = sum_j a*b
        // if (a[e] + pred[e] * b[e] * (field::GF2E(1) - b[e] * c[e]) != d[e])
        //     printf("pcp check failed\n");

        // field::GF2E accum_p;
        // field::GF2E accum_q;
        // for (size_t j = 0; j < instance.m1; j++) {
        //   accum_p += a[e][j] * (r_ejs[repetition][j] +  a[repetition][j] * b[repetition][j]);
        //   accum_q += b[repetition][j] * (r_ejs[repetition][j] +  a[repetition][j] * b[repetition][j]);
        // }
        // if (accum_p != c[repetition] || accum_q != d[repetition])
        //   throw std::runtime_error("final sanity check is wrong");
        proofs.push_back(proof);
    }

    dubhe_signature_t signature{salt, h_1, h_3, proofs};

    return signature;
}

bool dubhe_verify(const dubhe_instance_t &instance,
                    const std::vector<uint8_t> &pk,
                    const dubhe_signature_t &signature,
                    const uint8_t *message, size_t message_len) {
  // init modulus of extension field F_{2^{8\lambda}}
  // F::init_extension_field();
    field::GF2E::init_extension_field(instance);


  std::vector<uint8_t> pt(instance.aes_params.block_size *
                          instance.aes_params.num_blocks),
      ct(instance.aes_params.block_size * instance.aes_params.num_blocks);
  memcpy(pt.data(), pk.data(), pt.size());
  memcpy(ct.data(), pk.data() + pt.size(), ct.size());

  // do parallel repetitions
  // create seed trees and random tapes
  std::vector<SeedTree> seed_trees;
  seed_trees.reserve(instance.num_rounds);

  int bit_len = instance.aes_params.bit_len;
  size_t random_tape_size =
      instance.aes_params.key_size +
      instance.aes_params.num_sboxes +
      3 * bit_len * instance.lambda +
      2 * bit_len * instance.lambda +
      3 * instance.lambda + 3 * instance.lambda;
  RandomTapes random_tapes(instance.num_rounds, instance.num_MPC_parties,
                           random_tape_size);
  RepByteContainer party_seed_commitments(
      instance.num_rounds, instance.num_MPC_parties, instance.digest_size);

  // sumcheck
  std::vector<uint8_t> h_i = signature.h_1;

    std::vector<std::vector<std::vector<field::GF2E>>> coef1_deltas(bit_len);
    std::vector<std::vector<std::vector<field::GF2E>>> coef2_deltas(bit_len);
    std::vector<std::vector<field::GF2E>> rho;
    std::vector<std::vector<field::GF2E>> sgm;
    std::vector<std::vector<field::GF2E>> tau = phase_sumcheck0_expand(instance, h_i);
    std::vector<field::GF2E> pred(instance.num_rounds, field::GF2E(1));
    for (int i = 0; i < bit_len; i++) {
        coef1_deltas[i].resize(instance.num_rounds);
        for (size_t e = 0; e < instance.num_rounds; e++)
            coef1_deltas[i][e] = signature.proofs[e].coef1_delta[i];
        h_i = phase_sumcheck_commitment(instance, signature.salt, h_i, coef1_deltas[i]);
        std::vector<field::GF2E> rho_i = phase_sumcheck_expand(instance, h_i);
        rho.push_back(rho_i);
    }

    for (int i = 0; i < bit_len; i++) {
        coef2_deltas[i].resize(instance.num_rounds);
        for (size_t e = 0; e < instance.num_rounds; e++)
            coef2_deltas[i][e] = signature.proofs[e].coef2_delta[i];
        h_i = phase_sumcheck_commitment(instance, signature.salt, h_i, coef2_deltas[i]);
        std::vector<field::GF2E> sgm_i = phase_sumcheck_expand(instance, h_i);
        sgm.push_back(sgm_i);
    }
    for (size_t e = 0; e < instance.num_rounds; e++) {
        for (int i = 0; i < bit_len - 1; i++) {
            pred[e] *= tau[i][e] * rho[i][e] + (tau[i][e] + rho[i][e] + field::GF2E(1)) * (sgm[i][e] + field::GF2E(1));
        }
        pred[e] *= tau[bit_len-1][e] * rho[bit_len-1][e] + (tau[bit_len-1][e] + rho[bit_len-1][e] + field::GF2E(1)) * sgm[bit_len-1][e];
    }

  // recompute h_2
  std::vector<std::vector<field::GF2E>> P_deltas;
  for (const dubhe_repetition_proof_t &proof : signature.proofs) {
    P_deltas.push_back(proof.P_delta);
  }
  std::vector<uint8_t> h_2 =
      phase_2_commitment(instance, signature.salt, h_i, P_deltas);


  // compute challenges based on hashes

  // h2 expansion
  std::vector<field::GF2E> forbidden_challenge_values = field::get_first_n_field_elements(1);
  std::vector<field::GF2E> R_es = phase_2_expand(instance, h_2, forbidden_challenge_values);
  // h3 expansion already happened in deserialize to get missing parties
  std::vector<uint16_t> missing_parties = phase_3_expand(instance, signature.h_3);

  // rebuild SeedTrees
  for (size_t repetition = 0; repetition < instance.num_rounds; repetition++) {
    const dubhe_repetition_proof_t &proof = signature.proofs[repetition];
    // regenerate generate seed tree for the N parties (except the missing
    // one)
    if (missing_parties[repetition] != proof.reveallist.second)
      throw std::runtime_error(
          "modified signature between deserialization and verify");
    seed_trees.push_back(SeedTree(proof.reveallist, instance.num_MPC_parties,
                                  signature.salt, repetition));
    // commit to each party's seed, fill up missing one with data from proof
    {
      std::vector<uint8_t> dummy(instance.seed_size);
      size_t party = 0;
      for (; party < (instance.num_MPC_parties / 4) * 4; party += 4) {
        auto seed0 = seed_trees[repetition].get_leaf(party).value_or(dummy);
        auto seed1 = seed_trees[repetition].get_leaf(party + 1).value_or(dummy);
        auto seed2 = seed_trees[repetition].get_leaf(party + 2).value_or(dummy);
        auto seed3 = seed_trees[repetition].get_leaf(party + 3).value_or(dummy);
        commit_to_4_party_seeds(
            instance, seed0, seed1, seed2, seed3, signature.salt, repetition,
            party, party_seed_commitments.get(repetition, party),
            party_seed_commitments.get(repetition, party + 1),
            party_seed_commitments.get(repetition, party + 2),
            party_seed_commitments.get(repetition, party + 3));
      }
      for (; party < instance.num_MPC_parties; party++) {
        if (party != missing_parties[repetition]) {
          commit_to_party_seed(instance,
                               seed_trees[repetition].get_leaf(party).value(),
                               signature.salt, repetition, party,
                               party_seed_commitments.get(repetition, party));
        }
      }
    }
    auto com =
        party_seed_commitments.get(repetition, missing_parties[repetition]);
    std::copy(std::begin(proof.C_e), std::end(proof.C_e), std::begin(com));

    // create random tape for each party, dummy one for missing party
    {
      size_t party = 0;
      std::vector<uint8_t> dummy(instance.seed_size);
      for (; party < (instance.num_MPC_parties / 4) * 4; party += 4) {
        random_tapes.generate_4_tapes(
            repetition, party, signature.salt,
            seed_trees[repetition].get_leaf(party).value_or(dummy),
            seed_trees[repetition].get_leaf(party + 1).value_or(dummy),
            seed_trees[repetition].get_leaf(party + 2).value_or(dummy),
            seed_trees[repetition].get_leaf(party + 3).value_or(dummy));
      }
      for (; party < instance.num_MPC_parties; party++) {
        random_tapes.generate_tape(
            repetition, party, signature.salt,
            seed_trees[repetition].get_leaf(party).value_or(dummy));
      }
    }
  }

  /////////////////////////////////////////////////////////////////////////////
  // recompute commitments to executions of AES
  /////////////////////////////////////////////////////////////////////////////

  RepByteContainer rep_shared_keys(instance.num_rounds,
                                   instance.num_MPC_parties,
                                   instance.aes_params.key_size);
  RepByteContainer rep_shared_s(instance.num_rounds, instance.num_MPC_parties,
                                instance.aes_params.num_sboxes);
  RepByteContainer rep_shared_t(instance.num_rounds, instance.num_MPC_parties,
                                instance.aes_params.num_sboxes);
  RepByteContainer rep_output_broadcasts(
      instance.num_rounds, instance.num_MPC_parties,
      instance.aes_params.block_size * instance.aes_params.num_blocks);

  for (size_t repetition = 0; repetition < instance.num_rounds; repetition++) {
    const dubhe_repetition_proof_t &proof = signature.proofs[repetition];

    // generate sharing of secret key
    for (size_t party = 0; party < instance.num_MPC_parties; party++) {
      auto shared_key = rep_shared_keys.get(repetition, party);
      auto random_key_share =
          random_tapes.get_bytes(repetition, party, 0, shared_key.size());
      std::copy(std::begin(random_key_share), std::end(random_key_share),
                std::begin(shared_key));
    }

    // fix first share
    auto first_key_share = rep_shared_keys.get(repetition, 0);
    std::transform(std::begin(proof.sk_delta), std::end(proof.sk_delta),
                   std::begin(first_key_share), std::begin(first_key_share),
                   std::bit_xor<uint8_t>());

    // generate sharing of t values
    for (size_t party = 0; party < instance.num_MPC_parties; party++) {
      auto shared_t = rep_shared_t.get(repetition, party);
      auto random_t_shares = random_tapes.get_bytes(
          repetition, party, instance.aes_params.key_size,
          instance.aes_params.num_sboxes);
      std::copy(std::begin(random_t_shares), std::end(random_t_shares),
                std::begin(shared_t));
    }
    // fix first share
    auto first_shared_t = rep_shared_t.get(repetition, 0);
    std::transform(std::begin(proof.t_delta), std::end(proof.t_delta),
                   std::begin(first_shared_t), std::begin(first_shared_t),
                   std::bit_xor<uint8_t>());

    // get shares of sbox inputs by executing MPC AES
    auto ct_shares = rep_output_broadcasts.get_repetition(repetition);
    auto shared_s = rep_shared_s.get_repetition(repetition);

    if (instance.aes_params.key_size == 16)
      AES128::aes_128_s_shares(rep_shared_keys.get_repetition(repetition),
                               rep_shared_t.get_repetition(repetition), pt,
                               ct_shares, shared_s);
    else if (instance.aes_params.key_size == 24)
      AES192::aes_192_s_shares(rep_shared_keys.get_repetition(repetition),
                               rep_shared_t.get_repetition(repetition), pt,
                               ct_shares, shared_s);
    else if (instance.aes_params.key_size == 32)
      AES256::aes_256_s_shares(rep_shared_keys.get_repetition(repetition),
                               rep_shared_t.get_repetition(repetition), pt,
                               ct_shares, shared_s);
    else
      throw std::runtime_error("invalid parameters");

    // calculate missing output broadcast
    std::copy(ct.begin(), ct.end(),
              ct_shares[missing_parties[repetition]].begin());
    for (size_t party = 0; party < instance.num_MPC_parties; party++) {
      if (party != missing_parties[repetition])
        std::transform(std::begin(ct_shares[party]), std::end(ct_shares[party]),
                       std::begin(ct_shares[missing_parties[repetition]]),
                       std::begin(ct_shares[missing_parties[repetition]]),
                       std::bit_xor<uint8_t>());
    }
  }

  /////////////////////////////////////////////////////////////////////////////
  // recompute shares of polynomials
  /////////////////////////////////////////////////////////////////////////////

    // std::vector<std::vector<std::vector<field::GF2E>>> inputs_rho_share(instance.num_rounds);
    // std::vector<std::vector<std::vector<field::GF2E>>> inputs_sgm_share(instance.num_rounds);
    // RepContainer<std::array<field::GF2E, 3>> coef1_shares(instance.num_rounds, instance.num_MPC_parties, bit_len);
    // RepContainer<std::array<field::GF2E, 2>> coef2_shares(instance.num_rounds, instance.num_MPC_parties, bit_len);
    RepContainer<field::GF2E> Vr_share(instance.num_rounds, instance.num_MPC_parties, 2);
    RepContainer<field::GF2E> Vs_share(instance.num_rounds, instance.num_MPC_parties, 2);
    RepContainer<field::GF2E> fr_share(instance.num_rounds, instance.num_MPC_parties, 2);

    std::vector<std::vector<std::vector<uint8_t>>> inputs_rho_share_v(instance.num_rounds);
    std::vector<std::vector<std::vector<uint8_t>>> inputs_sgm_share_v(instance.num_rounds);
    std::vector<std::vector<std::array<std::vector<uint8_t>, 5>>> coef_shares_v(instance.num_rounds);
    std::vector<std::vector<uint8_t>> fr_shares_v(instance.num_rounds);

    for (size_t e = 0; e < instance.num_rounds; e++) {
        coef_shares_v[e].resize(bit_len);
        fr_shares_v[e].resize(instance.num_MPC_parties * 2);
        inputs_rho_share_v[e].resize(1 << bit_len);
        inputs_sgm_share_v[e].resize(1 << bit_len);
        for (int i = 0; i < bit_len; i++)
            for (int j = 0; j < 5; j++)
                coef_shares_v[e][i][j].resize(instance.num_MPC_parties * 2);
        for (int i = 0; i < (1 << bit_len); i++) {
            inputs_rho_share_v[e][i].resize(instance.num_MPC_parties * 2);
            inputs_sgm_share_v[e][i].resize(instance.num_MPC_parties * 2);
        }
        for (size_t i = 0; i < instance.aes_params.num_sboxes; i++) {
            size_t j = (1 << (bit_len - 1)) + i;
            for (size_t k = 0; k < instance.num_MPC_parties / 16; k++) {
                for (size_t p = 0; p < 16; p++) {
                    auto si = field::lift_uint8_t(rep_shared_s.get(e, k * 16 + p)[i]).data;
                    auto ti = field::lift_uint8_t(rep_shared_t.get(e, k * 16 + p)[i]).data;
                    inputs_rho_share_v[e][i][k * 32 + p] = si & 255;
                    inputs_rho_share_v[e][i][k * 32 + 16 + p] = si >> 8;
                    inputs_rho_share_v[e][j][k * 32 + p] = ti & 255;
                    inputs_rho_share_v[e][j][k * 32 + 16 + p] = ti >> 8;
                    inputs_sgm_share_v[e][i][k * 32 + 16 + p] = si >> 8;
                    inputs_sgm_share_v[e][i][k * 32 + p] = si & 255;
                    inputs_sgm_share_v[e][j][k * 32 + 16 + p] = ti >> 8;
                    inputs_sgm_share_v[e][j][k * 32 + p] = ti & 255;
                }
            }
        }
    }

    // for (size_t e = 0; e < instance.num_rounds; e++) {
    //     inputs_rho_share[e].resize(instance.num_MPC_parties);
    //     inputs_sgm_share[e].resize(instance.num_MPC_parties);
    //     for (size_t p = 0; p < instance.num_MPC_parties; p++) {
    //         inputs_rho_share[e][p].resize(1 << bit_len);
    //         inputs_sgm_share[e][p].resize(1 << bit_len);
    //         auto shared_s = rep_shared_s.get(e, p);
    //         auto shared_t = rep_shared_t.get(e, p);
    //         for (size_t i = 0; i < instance.aes_params.num_sboxes; i++) {
    //             size_t j = (1 << (bit_len - 1)) + i;
    //             inputs_rho_share[e][p][i] = field::lift_uint8_t(shared_s[i]);
    //             inputs_rho_share[e][p][j] = field::lift_uint8_t(shared_t[i]);
    //             inputs_sgm_share[e][p][i] = field::lift_uint8_t(shared_s[i]);
    //             inputs_sgm_share[e][p][j] = field::lift_uint8_t(shared_t[i]);
    //         }
    //     }
    // }
    for (int i = 0; i < bit_len; i++)
        for (size_t e = 0; e < instance.num_rounds; e++) {
            auto r = rho[i][e];
            auto r_raw = r.data;
            const __m256i table0 = _mm256_loadu_si256((__m256i *)&(field::GF2E::mul_lut[r_raw][0]));
            const __m256i table1 = _mm256_loadu_si256((__m256i *)&(field::GF2E::mul_lut[r_raw][8]));
            const __m256i table2 = _mm256_loadu_si256((__m256i *)&(field::GF2E::mul_lut[r_raw][16]));
            const __m256i table3 = _mm256_loadu_si256((__m256i *)&(field::GF2E::mul_lut[r_raw][24]));
            const __m256i mask1 = _mm256_set1_epi8(0x0f);
            const __m256i mask2 = _mm256_set1_epi8(0xf0);

            for (int u = 0; u < (1 << (bit_len - i - 1)); u++) {
                int u0 = u << 1;
                int u1 = u0 + 1;
                // for (size_t p = 0; p < instance.num_MPC_parties; p++)
                //     inputs_rho_share[e][p][u] =
                //         inputs_rho_share[e][p][u<<1] +
                //         (inputs_rho_share[e][p][(u<<1)+1] - inputs_rho_share[e][p][u<<1]) * rho[i][e];
                for (size_t k = 0; k < instance.num_MPC_parties / 16; k++) {
                    __m256i lo, hi, tmp;
                    __m256i t_u0 = _mm256_loadu_si256((__m256i *)&(inputs_rho_share_v[e][u0][k * 32]));
                    __m256i t_u1 = _mm256_loadu_si256((__m256i *)&(inputs_rho_share_v[e][u1][k * 32]));
                    t_u1 = _mm256_xor_si256(t_u0, t_u1);

                    // (t_u1 + t_u0) * r
                    lo = _mm256_and_si256(t_u1, mask1);  // a0, a2
                    lo = _mm256_shuffle_epi8(table0, lo);   // a0lo, a2hi
                    hi = _mm256_srli_epi64(_mm256_and_si256(t_u1, mask2), 4);    // a1, a3
                    hi = _mm256_shuffle_epi8(table1, hi);   // a1lo, a3hi
                    tmp = _mm256_xor_si256(hi, lo);
                    t_u1 = _mm256_permute4x64_epi64(t_u1, 2 | (3 << 2) | (1 << 6));   // a2,a3; a0,a1
                    lo = _mm256_and_si256(t_u1, mask1);  // a2, a0
                    lo = _mm256_shuffle_epi8(table2, lo);   // a2lo, a0hi
                    hi = _mm256_srli_epi64(_mm256_and_si256(t_u1, mask2), 4);    // a3, a1
                    hi = _mm256_shuffle_epi8(table3, hi);   // a3lo, a1hi
                    t_u1 = _mm256_xor_si256(tmp, _mm256_xor_si256(hi, lo));

                    // + t_u0
                    t_u1 = _mm256_xor_si256(t_u0, t_u1);
                    _mm256_storeu_si256((__m256i *)&(inputs_rho_share_v[e][u][k * 32]), t_u1);
                }
            }

            for (size_t k = 0; k < instance.num_MPC_parties / 16; k++) {
                for (size_t p = 0; p < 16; p++) {
                    auto random_coef_share =
                        random_tapes.get_bytes(e, k * 16 + p,
                                instance.aes_params.key_size + instance.aes_params.num_sboxes + 3 * i * instance.lambda,
                                3 * instance.lambda);
                    coef_shares_v[e][i][0][k * 32 + p] = *(uint8_t *)(random_coef_share.data());
                    coef_shares_v[e][i][1][k * 32 + p] = *(uint8_t *)(random_coef_share.data() + 2);
                    coef_shares_v[e][i][2][k * 32 + p] = *(uint8_t *)(random_coef_share.data() + 4);
                    coef_shares_v[e][i][0][k * 32 + 16 + p] = *(uint8_t *)(random_coef_share.data() + 1);
                    coef_shares_v[e][i][1][k * 32 + 16 + p] = *(uint8_t *)(random_coef_share.data() + 3);
                    coef_shares_v[e][i][2][k * 32 + 16 + p] = *(uint8_t *)(random_coef_share.data() + 5);
                }
            }
            coef_shares_v[e][i][0][0] ^= (coef1_deltas[i][e][0].data & 255);
            coef_shares_v[e][i][1][0] ^= (coef1_deltas[i][e][1].data & 255);
            coef_shares_v[e][i][2][0] ^= (coef1_deltas[i][e][2].data & 255);
            coef_shares_v[e][i][0][16] ^= (coef1_deltas[i][e][0].data >> 8);
            coef_shares_v[e][i][1][16] ^= (coef1_deltas[i][e][1].data >> 8);
            coef_shares_v[e][i][2][16] ^= (coef1_deltas[i][e][2].data >> 8);
            for (size_t k = 0; k < instance.num_MPC_parties / 16; k++) {
                __m256i lo, hi, tmp;
                __m256i t_coef0 = _mm256_loadu_si256((__m256i *)&(coef_shares_v[e][i][0][k * 32]));
                __m256i t_coef1 = _mm256_loadu_si256((__m256i *)&(coef_shares_v[e][i][1][k * 32]));
                __m256i t_coef2 = _mm256_loadu_si256((__m256i *)&(coef_shares_v[e][i][2][k * 32]));
                __m256i t_fr    = _mm256_loadu_si256((__m256i *)&(fr_shares_v[e][k * 32]));
                t_fr = _mm256_xor_si256(_mm256_xor_si256(t_coef1, t_coef2), t_fr);
                // coef[2] * r
                //
                lo = _mm256_and_si256(t_coef2, mask1);  // a0, a2
                lo = _mm256_shuffle_epi8(table0, lo);   // a0lo, a2hi
                hi = _mm256_srli_epi64(_mm256_and_si256(t_coef2, mask2), 4);    // a1, a3
                hi = _mm256_shuffle_epi8(table1, hi);   // a1lo, a3hi
                tmp = _mm256_xor_si256(hi, lo);
                t_coef2 = _mm256_permute4x64_epi64(t_coef2, 2 | (3 << 2) | (1 << 6));   // a2,a3; a0,a1
                lo = _mm256_and_si256(t_coef2, mask1);  // a2, a0
                lo = _mm256_shuffle_epi8(table2, lo);   // a2lo, a0hi
                hi = _mm256_srli_epi64(_mm256_and_si256(t_coef2, mask2), 4);    // a3, a1
                hi = _mm256_shuffle_epi8(table3, hi);   // a3lo, a1hi
                t_coef2 = _mm256_xor_si256(tmp, _mm256_xor_si256(hi, lo));

                // + coef[1]
                t_coef2 = _mm256_xor_si256(t_coef1, t_coef2);
                // * r
                lo = _mm256_and_si256(t_coef2, mask1);  // a0, a2
                lo = _mm256_shuffle_epi8(table0, lo);   // a0lo, a2hi
                hi = _mm256_srli_epi64(_mm256_and_si256(t_coef2, mask2), 4);    // a1, a3
                hi = _mm256_shuffle_epi8(table1, hi);   // a1lo, a3hi
                tmp = _mm256_xor_si256(hi, lo);
                t_coef2 = _mm256_permute4x64_epi64(t_coef2, 2 | (3 << 2) | (1 << 6));   // a2,a3; a0,a1
                lo = _mm256_and_si256(t_coef2, mask1);  // a2, a0
                lo = _mm256_shuffle_epi8(table2, lo);   // a2lo, a0hi
                hi = _mm256_srli_epi64(_mm256_and_si256(t_coef2, mask2), 4);    // a3, a1
                hi = _mm256_shuffle_epi8(table3, hi);   // a3lo, a1hi
                t_coef2 = _mm256_xor_si256(tmp, _mm256_xor_si256(hi, lo));
                // + fr
                t_coef2 = _mm256_xor_si256(t_fr, t_coef2);
                // * r
                lo = _mm256_and_si256(t_coef2, mask1);  // a0, a2
                lo = _mm256_shuffle_epi8(table0, lo);   // a0lo, a2hi
                hi = _mm256_srli_epi64(_mm256_and_si256(t_coef2, mask2), 4);    // a1, a3
                hi = _mm256_shuffle_epi8(table1, hi);   // a1lo, a3hi
                tmp = _mm256_xor_si256(hi, lo);
                t_coef2 = _mm256_permute4x64_epi64(t_coef2, 2 | (3 << 2) | (1 << 6));   // a2,a3; a0,a1
                lo = _mm256_and_si256(t_coef2, mask1);  // a2, a0
                lo = _mm256_shuffle_epi8(table2, lo);   // a2lo, a0hi
                hi = _mm256_srli_epi64(_mm256_and_si256(t_coef2, mask2), 4);    // a3, a1
                hi = _mm256_shuffle_epi8(table3, hi);   // a3lo, a1hi
                t_coef2 = _mm256_xor_si256(tmp, _mm256_xor_si256(hi, lo));
                // + coef[0]
                t_coef2 = _mm256_xor_si256(t_coef0, t_coef2);

                _mm256_storeu_si256((__m256i *)&(fr_shares_v[e][k * 32]), t_coef2);
            }

            // for (size_t p = 0; p < instance.num_MPC_parties; p++) {
            //     auto shared_coef = coef1_shares.get(e, p);
            //     auto fr = fr_share.get(e, p);
            //     auto random_coef_share = random_tapes.get_bytes(e, p,
            //                 instance.aes_params.key_size +
            //                 instance.aes_params.num_sboxes +
            //                 3 * i * instance.lambda,
            //                 3 * instance.lambda);
            //     shared_coef[i][0].from_bytes(random_coef_share.data());
            //     shared_coef[i][1].from_bytes(random_coef_share.data() + instance.lambda);
            //     shared_coef[i][2].from_bytes(random_coef_share.data() + instance.lambda * 2);
            //     if (p == 0) {
            //         shared_coef[i][0] += coef1_deltas[i][e][0];
            //         shared_coef[i][1] += coef1_deltas[i][e][1];
            //         shared_coef[i][2] += coef1_deltas[i][e][2];
            //     }
            //     fr[0] -= shared_coef[i][1] + shared_coef[i][2];
            //     fr[0] = ((shared_coef[i][2] * rho[i][e] + shared_coef[i][1]) * rho[i][e] + fr[0]) * rho[i][e] + shared_coef[i][0];
            // }

        }
    for (int i = 0; i < bit_len; i++)
        for (size_t e = 0; e < instance.num_rounds; e++) {
            auto r = sgm[i][e];
            auto r_raw = r.data;
            __m256i table0 = _mm256_loadu_si256((__m256i *)&(field::GF2E::mul_lut[r_raw][0]));
            __m256i table1 = _mm256_loadu_si256((__m256i *)&(field::GF2E::mul_lut[r_raw][8]));
            __m256i table2 = _mm256_loadu_si256((__m256i *)&(field::GF2E::mul_lut[r_raw][16]));
            __m256i table3 = _mm256_loadu_si256((__m256i *)&(field::GF2E::mul_lut[r_raw][24]));
            __m256i mask1 = _mm256_set1_epi8(0x0f);
            __m256i mask2 = _mm256_set1_epi8(0xf0);
            for (int u = 0; u < (1 << (bit_len - i - 1)); u++) {
                int u0 = u << 1;
                int u1 = u0 + 1;
                // for (size_t p = 0; p < instance.num_MPC_parties; p++)
                //     inputs_sgm_share[e][p][u] =
                //         inputs_sgm_share[e][p][u<<1] +
                //         (inputs_sgm_share[e][p][(u<<1)+1] - inputs_sgm_share[e][p][u<<1]) * sgm[i][e];
                for (size_t k = 0; k < instance.num_MPC_parties / 16; k++) {
                    __m256i lo, hi, tmp;
                    __m256i t_u0 = _mm256_loadu_si256((__m256i *)&(inputs_sgm_share_v[e][u0][k * 32]));
                    __m256i t_u1 = _mm256_loadu_si256((__m256i *)&(inputs_sgm_share_v[e][u1][k * 32]));
                    t_u1 = _mm256_xor_si256(t_u0, t_u1);

                    // (t_u1 + t_u0) * r
                    lo = _mm256_and_si256(t_u1, mask1);  // a0, a2
                    lo = _mm256_shuffle_epi8(table0, lo);   // a0lo, a2hi
                    hi = _mm256_srli_epi64(_mm256_and_si256(t_u1, mask2), 4);    // a1, a3
                    hi = _mm256_shuffle_epi8(table1, hi);   // a1lo, a3hi
                    tmp = _mm256_xor_si256(hi, lo);
                    t_u1 = _mm256_permute4x64_epi64(t_u1, 2 | (3 << 2) | (1 << 6));   // a2,a3; a0,a1
                    lo = _mm256_and_si256(t_u1, mask1);  // a2, a0
                    lo = _mm256_shuffle_epi8(table2, lo);   // a2lo, a0hi
                    hi = _mm256_srli_epi64(_mm256_and_si256(t_u1, mask2), 4);    // a3, a1
                    hi = _mm256_shuffle_epi8(table3, hi);   // a3lo, a1hi
                    t_u1 = _mm256_xor_si256(tmp, _mm256_xor_si256(hi, lo));

                    // + t_u0
                    t_u1 = _mm256_xor_si256(t_u0, t_u1);
                    _mm256_storeu_si256((__m256i *)&(inputs_sgm_share_v[e][u][k * 32]), t_u1);
                }
            }

            for (size_t k = 0; k < instance.num_MPC_parties / 16; k++) {
                for (size_t p = 0; p < 16; p++) {
                    auto random_coef_share =
                        random_tapes.get_bytes(e, k * 16 + p,
                                instance.aes_params.key_size +
                                instance.aes_params.num_sboxes +
                                3 * bit_len * instance.lambda +
                                2 * i * instance.lambda,
                                2 * instance.lambda);
                    coef_shares_v[e][i][3][k * 32 + p] = *(uint8_t *)(random_coef_share.data());
                    coef_shares_v[e][i][4][k * 32 + p] = *(uint8_t *)(random_coef_share.data() + 2);
                    coef_shares_v[e][i][3][k * 32 + 16 + p] = *(uint8_t *)(random_coef_share.data() + 1);
                    coef_shares_v[e][i][4][k * 32 + 16 + p] = *(uint8_t *)(random_coef_share.data() + 3);
                }
            }
            coef_shares_v[e][i][3][0] ^= (coef2_deltas[i][e][0].data & 255);
            coef_shares_v[e][i][4][0] ^= (coef2_deltas[i][e][1].data & 255);
            coef_shares_v[e][i][3][16] ^= (coef2_deltas[i][e][0].data >> 8);
            coef_shares_v[e][i][4][16] ^= (coef2_deltas[i][e][1].data >> 8);

            for (size_t k = 0; k < instance.num_MPC_parties / 16; k++) {
                __m256i lo, hi, tmp;
                __m256i t_coef0 = _mm256_loadu_si256((__m256i *)&(coef_shares_v[e][i][3][k * 32]));
                __m256i t_coef1 = _mm256_loadu_si256((__m256i *)&(coef_shares_v[e][i][4][k * 32]));
                __m256i t_fr    = _mm256_loadu_si256((__m256i *)&(fr_shares_v[e][k * 32]));
                t_fr = _mm256_xor_si256(t_coef1, t_fr);
                // coef1 * r
                lo = _mm256_and_si256(t_coef1, mask1);  // a0, a2
                lo = _mm256_shuffle_epi8(table0, lo);   // a0lo, a2hi
                hi = _mm256_srli_epi64(_mm256_and_si256(t_coef1, mask2), 4);    // a1, a3
                hi = _mm256_shuffle_epi8(table1, hi);   // a1lo, a3hi
                tmp = _mm256_xor_si256(hi, lo);
                t_coef1 = _mm256_permute4x64_epi64(t_coef1, 2 | (3 << 2) | (1 << 6));   // a2,a3; a0,a1
                lo = _mm256_and_si256(t_coef1, mask1);  // a2, a0
                lo = _mm256_shuffle_epi8(table2, lo);   // a2lo, a0hi
                hi = _mm256_srli_epi64(_mm256_and_si256(t_coef1, mask2), 4);    // a3, a1
                hi = _mm256_shuffle_epi8(table3, hi);   // a3lo, a1hi
                t_coef1 = _mm256_xor_si256(tmp, _mm256_xor_si256(hi, lo));
                // + fr
                t_coef1 = _mm256_xor_si256(t_fr, t_coef1);
                // * r
                lo = _mm256_and_si256(t_coef1, mask1);  // a0, a2
                lo = _mm256_shuffle_epi8(table0, lo);   // a0lo, a2hi
                hi = _mm256_srli_epi64(_mm256_and_si256(t_coef1, mask2), 4);    // a1, a3
                hi = _mm256_shuffle_epi8(table1, hi);   // a1lo, a3hi
                tmp = _mm256_xor_si256(hi, lo);
                t_coef1 = _mm256_permute4x64_epi64(t_coef1, 2 | (3 << 2) | (1 << 6));   // a2,a3; a0,a1
                lo = _mm256_and_si256(t_coef1, mask1);  // a2, a0
                lo = _mm256_shuffle_epi8(table2, lo);   // a2lo, a0hi
                hi = _mm256_srli_epi64(_mm256_and_si256(t_coef1, mask2), 4);    // a3, a1
                hi = _mm256_shuffle_epi8(table3, hi);   // a3lo, a1hi
                t_coef1 = _mm256_xor_si256(tmp, _mm256_xor_si256(hi, lo));
                // + coef[0]
                t_coef1 = _mm256_xor_si256(t_coef0, t_coef1);

                _mm256_storeu_si256((__m256i *)&(fr_shares_v[e][k * 32]), t_coef1);
            }

            // for (size_t p = 0; p < instance.num_MPC_parties; p++) {
            //     auto shared_coef = coef2_shares.get(e, p);
            //     auto fr = fr_share.get(e, p);
            //     auto random_coef_share = random_tapes.get_bytes(e, p,
            //                 instance.aes_params.key_size +
            //                 instance.aes_params.num_sboxes +
            //                 3 * bit_len * instance.lambda +
            //                 2 * i * instance.lambda,
            //                 2 * instance.lambda);
            //     shared_coef[i][0].from_bytes(random_coef_share.data());
            //     shared_coef[i][1].from_bytes(random_coef_share.data() + instance.lambda);
            //     if (p == 0) {
            //         shared_coef[i][0] += coef2_deltas[i][e][0];
            //         shared_coef[i][1] += coef2_deltas[i][e][1];
            //     }
            //     fr[0] -= shared_coef[i][1];
            //     fr[0] = (shared_coef[i][1] * sgm[i][e] + fr[0]) * sgm[i][e] + shared_coef[i][0];
            // }

        }
    for (size_t e = 0; e < instance.num_rounds; e++)
        for (size_t k = 0; k < instance.num_MPC_parties / 16; k++)
            for (size_t p = 0; p < 16; p++) {
                auto Vr = Vr_share.get(e, k * 16 + p);
                auto Vs = Vs_share.get(e, k * 16 + p);
                auto fr = fr_share.get(e, k * 16 + p);
                fr[0] = field::GF2E(fr_shares_v[e][k * 32 + p] | ((uint16_t)fr_shares_v[e][k * 32 + p + 16] << 8));
                Vr[0] = field::GF2E(inputs_rho_share_v[e][0][k * 32 + p] | ((uint16_t)inputs_rho_share_v[e][0][k * 32 + p + 16] << 8));
                Vs[0] = field::GF2E(inputs_sgm_share_v[e][0][k * 32 + p] | ((uint16_t)inputs_sgm_share_v[e][0][k * 32 + p + 16] << 8));
            }

    std::vector<field::GF2E> x_values_for_interpolation_zero_to_m2 = field::get_first_n_field_elements(1 + 1);
    std::vector<std::vector<field::GF2E>> precomputation_for_zero_to_m2 = precompute_lagrange_polynomials(x_values_for_interpolation_zero_to_m2);
    std::vector<field::GF2E> x_values_for_interpolation_zero_to_3m2 = field::get_first_n_field_elements(3 * 1 + 1);
    std::vector<std::vector<field::GF2E>> precomputation_for_zero_to_3m2 = precompute_lagrange_polynomials(x_values_for_interpolation_zero_to_3m2);

    std::vector<std::vector<std::vector<field::GF2E>>> fr_poly_share(instance.num_rounds);
    std::vector<std::vector<std::vector<field::GF2E>>> Vr_poly_share(instance.num_rounds);
    std::vector<std::vector<std::vector<field::GF2E>>> Vs_poly_share(instance.num_rounds);
    std::vector<std::vector<std::vector<field::GF2E>>> P_share(instance.num_rounds);
    for (size_t e = 0; e < instance.num_rounds; e++) {
        P_share[e].resize(instance.num_MPC_parties);
        fr_poly_share[e].resize(instance.num_MPC_parties);
        Vr_poly_share[e].resize(instance.num_MPC_parties);
        Vs_poly_share[e].resize(instance.num_MPC_parties);
        for (size_t p = 0; p < instance.num_MPC_parties; p++) {
            auto a = fr_share.get(e, p);
            auto b = Vr_share.get(e, p);
            auto c = Vs_share.get(e, p);
            auto random_share = random_tapes.get_bytes(e, p,
                        instance.aes_params.key_size +
                        instance.aes_params.num_sboxes +
                        3 * bit_len * instance.lambda +
                        2 * bit_len * instance.lambda,
                        3 * instance.lambda + 3 * instance.lambda);
            a[1].from_bytes(random_share.data());
            b[1].from_bytes(random_share.data() + instance.lambda);
            c[1].from_bytes(random_share.data() + 2 * instance.lambda);
            fr_poly_share[e][p] = interpolate_with_precomputation(precomputation_for_zero_to_m2, a);
            Vr_poly_share[e][p] = interpolate_with_precomputation(precomputation_for_zero_to_m2, b);
            Vs_poly_share[e][p] = interpolate_with_precomputation(precomputation_for_zero_to_m2, c);
            P_share[e][p].resize(3 * 1 + 1);
            P_share[e][p][1].from_bytes(random_share.data() + 3 * instance.lambda);
            P_share[e][p][2].from_bytes(random_share.data() + 4 * instance.lambda);
            P_share[e][p][3].from_bytes(random_share.data() + 5 * instance.lambda);
            if (p == 0)
                for (size_t k = 1; k <= 3 * 1; k++)
                    P_share[e][0][k] += P_deltas[e][k - 1];
        }
    }

    /////////////////////////////////////////////////////////////////////////////
    // recompute views of polynomial checks
    /////////////////////////////////////////////////////////////////////////////

    std::vector<std::vector<field::GF2E>> a_shares(instance.num_rounds);
    std::vector<std::vector<field::GF2E>> b_shares(instance.num_rounds);
    std::vector<std::vector<field::GF2E>> c_shares(instance.num_rounds);
    std::vector<std::vector<field::GF2E>> d_shares(instance.num_rounds);

    std::vector<field::GF2E> lagrange_polys_evaluated_at_Re_m2(1 + 1);
    std::vector<field::GF2E> lagrange_polys_evaluated_at_Re_3m2(3 * 1 + 1);

    for (size_t e = 0; e < instance.num_rounds; e++) {
        field::GF2E a = signature.proofs[e].fr_at_R;
        field::GF2E b = signature.proofs[e].Vr_at_R;
        field::GF2E c = signature.proofs[e].Vs_at_R;
        field::GF2E d = a + pred[e] * b * (field::GF2E(1) - b * c);
        a_shares[e].resize(instance.num_MPC_parties);
        b_shares[e].resize(instance.num_MPC_parties);
        c_shares[e].resize(instance.num_MPC_parties);
        d_shares[e].resize(instance.num_MPC_parties);
        for (size_t k = 0; k < 1 + 1; k++)
            lagrange_polys_evaluated_at_Re_m2[k] = eval(precomputation_for_zero_to_m2[k], R_es[e]);
        for (size_t k = 0; k < 3 * 1 + 1; k++)
            lagrange_polys_evaluated_at_Re_3m2[k] = eval(precomputation_for_zero_to_3m2[k], R_es[e]);

        for (size_t p = 0; p < instance.num_MPC_parties; p++)
            if (p != missing_parties[e]) {
                auto fr   = fr_share.get(e, p);
                auto Vr = Vr_share.get(e, p);
                auto Vs = Vs_share.get(e, p);
                a_shares[e][p] = dot_product(lagrange_polys_evaluated_at_Re_m2, fr);
                b_shares[e][p] = dot_product(lagrange_polys_evaluated_at_Re_m2, Vr);
                c_shares[e][p] = dot_product(lagrange_polys_evaluated_at_Re_m2, Vs);
                d_shares[e][p] = dot_product(lagrange_polys_evaluated_at_Re_3m2, P_share[e][p]);
                a -= a_shares[e][p];
                b -= b_shares[e][p];
                c -= c_shares[e][p];
                d -= d_shares[e][p];
            }
        a_shares[e][missing_parties[e]] = a;
        b_shares[e][missing_parties[e]] = b;
        c_shares[e][missing_parties[e]] = c;
        d_shares[e][missing_parties[e]] = d;
    }


    /////////////////////////////////////////////////////////////////////////////
    // recompute h_1 and h_3
    /////////////////////////////////////////////////////////////////////////////

    std::vector<std::vector<uint8_t>> sk_deltas;
    std::vector<std::vector<uint8_t>> t_deltas;
    std::vector<field::GF2E> a;
    std::vector<field::GF2E> b;
    std::vector<field::GF2E> c;
    for (const dubhe_repetition_proof_t &proof : signature.proofs) {
        sk_deltas.push_back(proof.sk_delta);
        t_deltas.push_back(proof.t_delta);
        a.push_back(proof.fr_at_R);
        b.push_back(proof.Vr_at_R);
        c.push_back(proof.Vs_at_R);
    }
    std::vector<uint8_t> h_1 = phase_1_commitment(
            instance, signature.salt,
            party_seed_commitments, sk_deltas) {

    std::vector<uint8_t> h_3 = phase_3_commitment(instance, signature.salt, h_2, d_shares, a, a_shares, b, b_shares, c, c_shares);

    if (memcmp(h_1.data(), signature.h_1.data(), h_1.size()) != 0) {
        return false;
    }
    if (memcmp(h_3.data(), signature.h_3.data(), h_3.size()) != 0) {
        return false;
    }

    return true;
}

std::vector<uint8_t>
dubhe_serialize_signature(const dubhe_instance_t &instance,
                            const dubhe_signature_t &signature) {
  std::vector<uint8_t> serialized;

  serialized.insert(serialized.end(), signature.salt.begin(),
                    signature.salt.end());
  serialized.insert(serialized.end(), signature.h_1.begin(),
                    signature.h_1.end());
  serialized.insert(serialized.end(), signature.h_3.begin(),
                    signature.h_3.end());

  for (size_t repetition = 0; repetition < instance.num_rounds; repetition++) {
    const dubhe_repetition_proof_t &proof = signature.proofs[repetition];
    for (const std::vector<uint8_t> &seed : proof.reveallist.first) {
      serialized.insert(serialized.end(), seed.begin(), seed.end());
    }
    serialized.insert(serialized.end(), proof.C_e.begin(), proof.C_e.end());
    serialized.insert(serialized.end(), proof.sk_delta.begin(),
                      proof.sk_delta.end());
    serialized.insert(serialized.end(), proof.t_delta.begin(),
                      proof.t_delta.end());
    for (size_t k = 0; k < instance.aes_params.bit_len; k++) {
        for (int i = 0; i < 3; i++) {
            std::vector<uint8_t> buffer = proof.coef1_delta[k][i].to_bytes();
            serialized.insert(serialized.end(), buffer.begin(), buffer.end());
        }
    }
    for (size_t k = 0; k < instance.aes_params.bit_len; k++) {
        for (int i = 0; i < 2; i++) {
            std::vector<uint8_t> buffer = proof.coef2_delta[k][i].to_bytes();
            serialized.insert(serialized.end(), buffer.begin(), buffer.end());
        }
    }
    for (size_t k = 0; k < 2 * 1 + 1; k++) {
      std::vector<uint8_t> buffer = proof.P_delta[k].to_bytes();
      serialized.insert(serialized.end(), buffer.begin(), buffer.end());
    }
    {
      std::vector<uint8_t> buffer = proof.fr_at_R.to_bytes();
      serialized.insert(serialized.end(), buffer.begin(), buffer.end());
    }
    {
      std::vector<uint8_t> buffer = proof.Vr_at_R.to_bytes();
      serialized.insert(serialized.end(), buffer.begin(), buffer.end());
    }
    {
      std::vector<uint8_t> buffer = proof.Vs_at_R.to_bytes();
      serialized.insert(serialized.end(), buffer.begin(), buffer.end());
    }
  }
  return serialized;
}

dubhe_signature_t
dubhe_deserialize_signature(const dubhe_instance_t &instance,
                              const std::vector<uint8_t> &serialized) {

  size_t current_offset = 0;
  dubhe_salt_t salt{};
  memcpy(salt.data(), serialized.data() + current_offset, salt.size());
  current_offset += salt.size();
  std::vector<uint8_t> h_1(instance.digest_size), h_3(instance.digest_size);
  memcpy(h_1.data(), serialized.data() + current_offset, h_1.size());
  current_offset += h_1.size();
  memcpy(h_3.data(), serialized.data() + current_offset, h_3.size());
  current_offset += h_3.size();
  std::vector<dubhe_repetition_proof_t> proofs;

  std::vector<uint16_t> missing_parties = phase_3_expand(instance, h_3);
  size_t reveallist_size = ceil_log2(instance.num_MPC_parties);
  for (size_t repetition = 0; repetition < instance.num_rounds; repetition++) {
    reveal_list_t reveallist;
    reveallist.first.reserve(reveallist_size);
    reveallist.second = missing_parties[repetition];
    for (size_t i = 0; i < reveallist_size; i++) {
      std::vector<uint8_t> seed(instance.seed_size);
      memcpy(seed.data(), serialized.data() + current_offset, seed.size());
      current_offset += seed.size();
      reveallist.first.push_back(seed);
    }
    std::vector<uint8_t> C_e(instance.digest_size);
    memcpy(C_e.data(), serialized.data() + current_offset, C_e.size());
    current_offset += C_e.size();

    std::vector<uint8_t> sk_delta(instance.aes_params.key_size);
    memcpy(sk_delta.data(), serialized.data() + current_offset,
           sk_delta.size());
    current_offset += sk_delta.size();

    std::vector<uint8_t> t_delta(instance.aes_params.num_sboxes);
    memcpy(t_delta.data(), serialized.data() + current_offset, t_delta.size());
    current_offset += t_delta.size();

    field::GF2E tmp;
    std::vector<std::vector<field::GF2E>> coef1_delta;
    coef1_delta.resize(instance.aes_params.bit_len);
    for (size_t k = 0; k < instance.aes_params.bit_len; k++) {
        for (size_t i = 0; i < 3; i++) {
          std::vector<uint8_t> buffer(instance.lambda);
          memcpy(buffer.data(), serialized.data() + current_offset, buffer.size());
          current_offset += buffer.size();
          tmp.from_bytes(buffer.data());
          coef1_delta[k].push_back(tmp);
        }
    }
    std::vector<std::vector<field::GF2E>> coef2_delta;
    coef2_delta.resize(instance.aes_params.bit_len);
    for (size_t k = 0; k < instance.aes_params.bit_len; k++) {
        for (size_t i = 0; i < 2; i++) {
          std::vector<uint8_t> buffer(instance.lambda);
          memcpy(buffer.data(), serialized.data() + current_offset, buffer.size());
          current_offset += buffer.size();
          tmp.from_bytes(buffer.data());
          coef2_delta[k].push_back(tmp);
        }
    }

    std::vector<field::GF2E> P_delta;
    P_delta.reserve(2 * 1 + 1);
    for (size_t k = 0; k < 2 * 1 + 1; k++) {
      std::vector<uint8_t> buffer(instance.lambda);
      memcpy(buffer.data(), serialized.data() + current_offset, buffer.size());
      current_offset += buffer.size();
      tmp.from_bytes(buffer.data());
      P_delta.push_back(tmp);
    }
    field::GF2E fr_at_R;
    {
      std::vector<uint8_t> buffer(instance.lambda);
      memcpy(buffer.data(), serialized.data() + current_offset, buffer.size());
      current_offset += buffer.size();
      fr_at_R.from_bytes(buffer.data());
    }
    field::GF2E Vr_at_R;
    {
      std::vector<uint8_t> buffer(instance.lambda);
      memcpy(buffer.data(), serialized.data() + current_offset, buffer.size());
      current_offset += buffer.size();
      Vr_at_R.from_bytes(buffer.data());
    }
    field::GF2E Vs_at_R;
    {
      std::vector<uint8_t> buffer(instance.lambda);
      memcpy(buffer.data(), serialized.data() + current_offset, buffer.size());
      current_offset += buffer.size();
      Vs_at_R.from_bytes(buffer.data());
    }
    proofs.emplace_back(dubhe_repetition_proof_t{reveallist, C_e, sk_delta,
                                                   t_delta, coef1_delta, coef2_delta, P_delta,
                                                   fr_at_R, Vr_at_R, Vs_at_R});
  }
  assert(current_offset == serialized.size());
  dubhe_signature_t signature{salt, h_1, h_3, proofs};

  return signature;
}

/* dubhe_group_signature_t dubhe_group_sign(
        const dubhe_instance_t &instance,
        const dubhe_keypair_t &keypair,
        const std::vector<dubhe_keypair_t> &gkey,
        const uint8_t *message, size_t message_len) {
  // init modulus of extension field F_{2^{8\lambda}}
    field::GF2E::init_extension_field(instance);

#ifdef TIMING
    timing_context_t ctx;
    timing_init(&ctx);

    uint64_t start_time = timing_read(&ctx);
    uint64_t tmp_time;
#endif

  // grab aes key, pt and ct
  std::vector<uint8_t> key = keypair.first;
  std::vector<uint8_t> pt_ct = keypair.second;
  const size_t total_pt_ct_size =
      instance.aes_params.block_size * instance.aes_params.num_blocks;
  std::vector<uint8_t> pt(total_pt_ct_size), ct(total_pt_ct_size),
      ct2(total_pt_ct_size);
  memcpy(pt.data(), keypair.second.data(), pt.size());
  memcpy(ct.data(), keypair.second.data() + pt.size(), ct.size());

  // get sbox inputs and outputs for aes evaluation
  std::pair<std::vector<uint8_t>, std::vector<uint8_t>> sbox_pairs;

  if (instance.aes_params.key_size == 16)
    sbox_pairs = AES128::aes_128_with_sbox_output(key, pt, ct2);
  else if (instance.aes_params.key_size == 24)
    sbox_pairs = AES192::aes_192_with_sbox_output(key, pt, ct2);
  else if (instance.aes_params.key_size == 32)
    sbox_pairs = AES256::aes_256_with_sbox_output(key, pt, ct2);
  else
    throw std::runtime_error("invalid parameters");
  // sanity check, incoming keypair is valid
  // assert(ct == ct2);

  // generate salt and master seeds for each repetition
  auto [salt, master_seeds] =
      generate_salt_and_seeds(instance);

  // do parallel repetitions
  // create seed trees and random tapes
  std::vector<SeedTree> seed_trees;
  seed_trees.reserve(instance.num_rounds);

  int bit_len = instance.aes_params.bit_len;
  int depth = ceil_log2(gkey.size());
  size_t random_tape_size =
      instance.aes_params.key_size +
      instance.aes_params.block_size * instance.aes_params.num_blocks +
      instance.aes_params.num_sboxes +
      3 * bit_len * instance.lambda +
      2 * bit_len * instance.lambda +
      3 * instance.lambda + 3 * instance.lambda +
      2 * (depth - 1) * instance.lambda +
      2 * depth * (depth + 1) * instance.lambda +
      (4 + depth) * instance.lambda;


  RandomTapes random_tapes(instance.num_rounds, instance.num_MPC_parties,
                           random_tape_size);

  RepByteContainer party_seed_commitments(
      instance.num_rounds, instance.num_MPC_parties, instance.digest_size);

  for (size_t repetition = 0; repetition < instance.num_rounds; repetition++) {
    // generate seed tree for the N parties
    seed_trees.emplace_back(master_seeds[repetition], instance.num_MPC_parties,
                            salt, repetition);

    // commit to each party's seed;
    {
      size_t party = 0;
      for (; party < (instance.num_MPC_parties / 4) * 4; party += 4) {
        commit_to_4_party_seeds(
            instance, seed_trees[repetition].get_leaf(party).value(),
            seed_trees[repetition].get_leaf(party + 1).value(),
            seed_trees[repetition].get_leaf(party + 2).value(),
            seed_trees[repetition].get_leaf(party + 3).value(), salt,
            repetition, party, party_seed_commitments.get(repetition, party),
            party_seed_commitments.get(repetition, party + 1),
            party_seed_commitments.get(repetition, party + 2),
            party_seed_commitments.get(repetition, party + 3));
      }
      for (; party < instance.num_MPC_parties; party++) {
        commit_to_party_seed(
            instance, seed_trees[repetition].get_leaf(party).value(), salt,
            repetition, party, party_seed_commitments.get(repetition, party));
      }
    }

    // create random tape for each party
    // for (size_t party = 0; party < instance.num_MPC_parties; party++) {
    //   random_tapes.generate_tape(
    //       repetition, party, salt,
    //       seed_trees[repetition].get_leaf(party).value());
    // }
    {
        size_t party = 0;
        for (; party < (instance.num_MPC_parties / 4) * 4; party += 4) {
            random_tapes.generate_4_tapes(
                    repetition, party, salt,
                    seed_trees[repetition].get_leaf(party).value(),
                    seed_trees[repetition].get_leaf(party + 1).value(),
                    seed_trees[repetition].get_leaf(party + 2).value(),
                    seed_trees[repetition].get_leaf(party + 3).value());
        }
        for (; party < instance.num_MPC_parties; party++) {
            random_tapes.generate_tape(
                    repetition, party, salt,
                    seed_trees[repetition].get_leaf(party).value());
        }
    }
  }
  /////////////////////////////////////////////////////////////////////////////
  // phase 1: commit to executions of AES
  /////////////////////////////////////////////////////////////////////////////
  RepByteContainer rep_shared_keys(instance.num_rounds,
                                   instance.num_MPC_parties,
                                   instance.aes_params.key_size);

  RepByteContainer rep_shared_pt(instance.num_rounds,
                                 instance.num_MPC_parties,
                                 instance.aes_params.block_size * instance.aes_params.num_blocks);
  RepByteContainer rep_shared_ct(instance.num_rounds,
                                 instance.num_MPC_parties,
                                 instance.aes_params.block_size * instance.aes_params.num_blocks);

  // RepByteContainer rep_output_broadcasts(
  //     instance.num_rounds, instance.num_MPC_parties,
  //     instance.aes_params.block_size * instance.aes_params.num_blocks);

  RepByteContainer rep_shared_s(instance.num_rounds, instance.num_MPC_parties,
                                instance.aes_params.num_sboxes);
  RepByteContainer rep_shared_t(instance.num_rounds, instance.num_MPC_parties,
                                instance.aes_params.num_sboxes);
  std::vector<std::vector<uint8_t>> rep_key_deltas;
  std::vector<std::vector<uint8_t>> rep_pt_deltas;
  std::vector<std::vector<uint8_t>> rep_t_deltas;

  for (size_t repetition = 0; repetition < instance.num_rounds; repetition++) {

    // generate sharing of secret key
    std::vector<uint8_t> key_delta = key;
    for (size_t party = 0; party < instance.num_MPC_parties; party++) {
      auto shared_key = rep_shared_keys.get(repetition, party);
      auto random_key_share =
          random_tapes.get_bytes(repetition, party, 0, shared_key.size());
      std::copy(std::begin(random_key_share), std::end(random_key_share),
                std::begin(shared_key));

      std::transform(std::begin(shared_key), std::end(shared_key),
                     std::begin(key_delta), std::begin(key_delta),
                     std::bit_xor<uint8_t>());
    }

    // fix first share
    auto first_share_key = rep_shared_keys.get(repetition, 0);
    std::transform(std::begin(key_delta), std::end(key_delta),
                   std::begin(first_share_key), std::begin(first_share_key),
                   std::bit_xor<uint8_t>());

    rep_key_deltas.push_back(key_delta);

    // generate sharing of pt
    std::vector<uint8_t> pt_delta = pt;
    for (size_t party = 0; party < instance.num_MPC_parties; party++) {
      auto shared_pt = rep_shared_pt.get(repetition, party);
      auto random_pt_share =
          random_tapes.get_bytes(repetition, party, instance.aes_params.key_size, shared_pt.size());
      std::copy(std::begin(random_pt_share), std::end(random_pt_share),
                std::begin(shared_pt));

      std::transform(std::begin(shared_pt), std::end(shared_pt),
                     std::begin(pt_delta), std::begin(pt_delta),
                     std::bit_xor<uint8_t>());
    }

    // fix first share
    auto first_share_pt = rep_shared_pt.get(repetition, 0);
    std::transform(std::begin(pt_delta), std::end(pt_delta),
                   std::begin(first_share_pt), std::begin(first_share_pt),
                   std::bit_xor<uint8_t>());

    rep_pt_deltas.push_back(pt_delta);


    // generate sharing of t values
    std::vector<uint8_t> t_deltas = sbox_pairs.second;
    for (size_t party = 0; party < instance.num_MPC_parties; party++) {
      auto shared_t = rep_shared_t.get(repetition, party);
      auto random_t_shares = random_tapes.get_bytes(
          repetition, party, instance.aes_params.key_size + instance.aes_params.block_size * instance.aes_params.num_blocks,
          instance.aes_params.num_sboxes);
      std::copy(std::begin(random_t_shares), std::end(random_t_shares),
                std::begin(shared_t));
      std::transform(std::begin(shared_t), std::end(shared_t),
                     std::begin(t_deltas), std::begin(t_deltas),
                     std::bit_xor<uint8_t>());
    }
    // fix first share
    auto first_share_t = rep_shared_t.get(repetition, 0);
    std::transform(std::begin(t_deltas), std::end(t_deltas),
                   std::begin(first_share_t), std::begin(first_share_t),
                   std::bit_xor<uint8_t>());

    // get shares of sbox inputs by executing MPC AES
    // auto ct_shares = rep_output_broadcasts.get_repetition(repetition);
    auto pt_shares = rep_shared_pt.get_repetition(repetition);
    auto ct_shares = rep_shared_ct.get_repetition(repetition);
    auto shared_s = rep_shared_s.get_repetition(repetition);

    if (instance.aes_params.key_size == 16)
      // AES128::aes_128_s_shares(rep_shared_keys.get_repetition(repetition),
      //                          rep_shared_t.get_repetition(repetition), pt,
      //                          ct_shares, shared_s);
      AES128::group_aes_128_s_shares(
              rep_shared_keys.get_repetition(repetition),
              rep_shared_t.get_repetition(repetition),
              pt_shares, ct_shares, shared_s);
    else if (instance.aes_params.key_size == 24)
      // AES192::aes_192_s_shares(rep_shared_keys.get_repetition(repetition),
      //                          rep_shared_t.get_repetition(repetition), pt,
      //                          ct_shares, shared_s);
      AES192::group_aes_192_s_shares(
              rep_shared_keys.get_repetition(repetition),
              rep_shared_t.get_repetition(repetition),
              pt_shares, ct_shares, shared_s);
    else if (instance.aes_params.key_size == 32)
      // AES256::aes_256_s_shares(rep_shared_keys.get_repetition(repetition),
      //                          rep_shared_t.get_repetition(repetition), pt,
      //                          ct_shares, shared_s);
      AES256::group_aes_256_s_shares(
              rep_shared_keys.get_repetition(repetition),
              rep_shared_t.get_repetition(repetition),
              pt_shares, ct_shares, shared_s);
    else
      throw std::runtime_error("invalid parameters");

#ifndef NDEBUG
    // sanity check, mpc execution = plain one
    // std::vector<uint8_t> ct_check(instance.aes_params.block_size *
    //                               instance.aes_params.num_blocks);
    // memset(ct_check.data(), 0, ct_check.size());
    // for (size_t party = 0; party < instance.num_MPC_parties; party++) {
    //   std::transform(std::begin(ct_shares[party]), std::end(ct_shares[party]),
    //                  std::begin(ct_check), std::begin(ct_check),
    //                  std::bit_xor<uint8_t>());
    // }

    // assert(ct == ct_check);
#endif
    rep_t_deltas.push_back(t_deltas);
  }

  /////////////////////////////////////////////////////////////////////////////
  // phase 2: challenge the multiplications
  /////////////////////////////////////////////////////////////////////////////


#ifdef TIMING
    tmp_time = timing_read(&ctx);
    printf("phase1 time: %ld\n", tmp_time - start_time);
    start_time = timing_read(&ctx);
#endif


    // commit to salt, (all commitments of parties seeds, key_delta, t_delta)
    // for all repetitions
    std::vector<uint8_t> h_1 =
        group_phase_1_commitment(instance, salt, message, message_len,
                           party_seed_commitments, rep_key_deltas, rep_t_deltas,
                           rep_pt_deltas);

    std::vector<std::vector<field::GF2E>> tau = phase_sumcheck0_expand(instance, h_1);

    std::vector<uint8_t> h_i = h_1;

    RepContainer<field::GF2E> Vrs_shares(instance.num_rounds, instance.num_MPC_parties, 2);
    // for polynomials, a[0] = fr, b[0] = Vr, c[0] = Vs
    RepContainer<field::GF2E> fr_share(instance.num_rounds, instance.num_MPC_parties, 2);
    RepContainer<field::GF2E> Vr_share(instance.num_rounds, instance.num_MPC_parties, 2);
    RepContainer<field::GF2E> Vs_share(instance.num_rounds, instance.num_MPC_parties, 2);

    // sumcheck init
    std::vector<std::vector<uint16_t>> Tau(instance.num_rounds);
    std::vector<std::vector<uint16_t>> Rho(instance.num_rounds);
    std::vector<std::vector<uint16_t>> inputs(instance.num_rounds);

    // std::vector<std::vector<uint32_t>> Tau_v(instance.num_rounds);
    // std::vector<std::vector<uint32_t>> Rho_v(instance.num_rounds);
    // std::vector<std::vector<uint32_t>> inputs_v(instance.num_rounds);

    // std::vector<std::vector<std::vector<field::GF2E>>> inputs_rho_share(instance.num_rounds);
    // std::vector<std::vector<std::vector<field::GF2E>>> inputs_sgm_share(instance.num_rounds);

    std::vector<std::vector<std::vector<uint8_t>>> inputs_rho_share_v(instance.num_rounds);
    std::vector<std::vector<std::vector<uint8_t>>> inputs_sgm_share_v(instance.num_rounds);

    std::vector<std::vector<uint16_t>> AG(instance.num_rounds);
    std::vector<std::vector<uint16_t>> AX(instance.num_rounds);
    std::vector<std::vector<uint16_t>> AV(instance.num_rounds);

    // std::vector<std::vector<uint32_t>> AG_v(instance.num_rounds);
    // std::vector<std::vector<uint32_t>> AX_v(instance.num_rounds);
    // std::vector<std::vector<uint32_t>> AV_v(instance.num_rounds);

    std::vector<std::vector<std::array<field::GF2E, 3>>> coef1(bit_len);
    std::vector<std::vector<std::vector<field::GF2E>>> coef1_deltas;
    std::vector<std::vector<std::array<field::GF2E, 2>>> coef2(bit_len);
    std::vector<std::vector<std::vector<field::GF2E>>> coef2_deltas;
    // the second element is for the sum of each party's random
    std::vector<std::array<field::GF2E, 2>> Vr(instance.num_rounds);
    std::vector<std::array<field::GF2E, 2>> Vs(instance.num_rounds);
    std::vector<std::array<field::GF2E, 2>> fr(instance.num_rounds);
    std::vector<std::vector<field::GF2E>> rho;
    std::vector<field::GF2E> pred(instance.num_rounds, field::GF2E(1));

    // std::vector<std::vector<std::array<std::vector<field::GF2E>, 5>>> coef_shares(instance.num_rounds);
    // std::vector<std::vector<field::GF2E>> fr_shares(instance.num_rounds);

    std::vector<std::vector<std::array<std::vector<uint8_t>, 5>>> coef_shares_v(instance.num_rounds);
    std::vector<std::vector<uint8_t>> fr_shares_v(instance.num_rounds);

#ifdef TIMING
    tmp_time = timing_read(&ctx);
    printf("gkr init declare time: %ld\n", tmp_time - start_time);
    start_time = timing_read(&ctx);
#endif

    for (size_t e = 0; e < instance.num_rounds; e++) {
        // coef_shares[e].resize(bit_len);
        coef_shares_v[e].resize(bit_len);
        for (int i = 0; i < bit_len; i++)
            for (int j = 0; j < 5; j++) {
                // coef_shares[e][i][j].resize(instance.num_MPC_parties);
                coef_shares_v[e][i][j].resize(instance.num_MPC_parties * 2);
                // for (size_t p = 0; p < instance.num_MPC_parties * 2; p++)
                //     coef_shares_v[e][i][j][p] = 0;
            }
        // fr_shares[e].resize(instance.num_MPC_parties);
        fr_shares_v[e].resize(instance.num_MPC_parties * 2);
        // for (size_t p = 0; p < instance.num_MPC_parties * 2; p++)
        //     fr_shares_v[e][p] = 0;
    }

#ifdef TIMING
    tmp_time = timing_read(&ctx);
    printf("gkr init vectors time: %ld\n", tmp_time - start_time);
    start_time = timing_read(&ctx);
    uint64_t ttt;
    uint64_t init_time1 = 0;
    uint64_t init_time2 = 0;
#endif

    for (size_t e = 0; e < instance.num_rounds; e++) {
#ifdef TIMING
    ttt = timing_read(&ctx);
#endif
        // inputs[e].resize(1 << bit_len);
        inputs[e].resize(1 << bit_len);
        // inputs_rho_share[e].resize(1 << bit_len);
        // inputs_sgm_share[e].resize(1 << bit_len);

        inputs_rho_share_v[e].resize(1 << bit_len);
        inputs_sgm_share_v[e].resize(1 << bit_len);

        for (size_t i = 0; i < instance.aes_params.num_sboxes; i++) {
            int j = (1 << (bit_len - 1)) + i;
            inputs[e][i] = field::lift(sbox_pairs.first[i]);
            inputs[e][j] = field::lift(sbox_pairs.second[i]);
            // inputs[e][i] = field::GF2E(sbox_pairs.first[i]);
            // inputs[e][j] = field::GF2E(sbox_pairs.second[i]);
            // inputs_v[e][i] = sbox_pairs.first[i];
            // inputs_v[e][j] = sbox_pairs.second[i];
        }
        for (int i = 0; i < (1 << bit_len); i++) {
            // inputs_rho_share[e][i].resize(instance.num_MPC_parties);
            // inputs_sgm_share[e][i].resize(instance.num_MPC_parties);
            inputs_rho_share_v[e][i].resize(instance.num_MPC_parties * 2);
            inputs_sgm_share_v[e][i].resize(instance.num_MPC_parties * 2);
        }
#ifdef TIMING
    init_time1 += timing_read(&ctx) - ttt;
    ttt = timing_read(&ctx);
#endif
        // for (size_t p = 0; p < instance.num_MPC_parties; p++) {
        //     auto shared_s = rep_shared_s.get(e, p);
        //     auto shared_t = rep_shared_t.get(e, p);
        //     for (size_t i = 0; i < instance.aes_params.num_sboxes; i++) {
        //         size_t j = (1 << (bit_len - 1)) + i;
        //         inputs_rho_share[e][i][p] = field::lift_uint8_t(shared_s[i]);
        //         inputs_rho_share[e][j][p] = field::lift_uint8_t(shared_t[i]);
        //         inputs_sgm_share[e][i][p] = field::lift_uint8_t(shared_s[i]);
        //         inputs_sgm_share[e][j][p] = field::lift_uint8_t(shared_t[i]);
        //     }
        // }

        for (size_t i = 0; i < instance.aes_params.num_sboxes; i++) {
            size_t j = (1 << (bit_len - 1)) + i;
            for (size_t k = 0; k < instance.num_MPC_parties / 16; k++) {
                for (size_t p = 0; p < 16; p++) {
                    // auto si = rep_shared_s.get(e, k * 16 + p)[i];
                    // auto ti = rep_shared_t.get(e, k * 16 + p)[i];
                    // inputs_rho_share_v[e][i][k * 32 + p] = si;
                    // inputs_rho_share_v[e][j][k * 32 + p] = ti;
                    // inputs_sgm_share_v[e][i][k * 32 + p] = si;
                    // inputs_sgm_share_v[e][j][k * 32 + p] = ti;

                    auto si = field::lift_uint8_t(rep_shared_s.get(e, k * 16 + p)[i]).data;
                    auto ti = field::lift_uint8_t(rep_shared_t.get(e, k * 16 + p)[i]).data;
                    inputs_rho_share_v[e][i][k * 32 + p] = si & 255;
                    inputs_rho_share_v[e][i][k * 32 + 16 + p] = si >> 8;
                    inputs_rho_share_v[e][j][k * 32 + p] = ti & 255;
                    inputs_rho_share_v[e][j][k * 32 + 16 + p] = ti >> 8;
                    inputs_sgm_share_v[e][i][k * 32 + 16 + p] = si >> 8;
                    inputs_sgm_share_v[e][i][k * 32 + p] = si & 255;
                    inputs_sgm_share_v[e][j][k * 32 + 16 + p] = ti >> 8;
                    inputs_sgm_share_v[e][j][k * 32 + p] = ti & 255;
                }
            }
        }
#ifdef TIMING
    init_time2 += timing_read(&ctx) - ttt;
#endif
    }
    // AV = inputs;
    AV = inputs;

#ifdef TIMING
    tmp_time = timing_read(&ctx);
    printf("gkr init AV time: %ld\n", tmp_time - start_time);
    printf("gkr init AV time1: %ld\n", init_time1);
    printf("gkr init AV time2: %ld\n", init_time2);
    start_time = timing_read(&ctx);
#endif

    // sumcheck phase 1 init
    for (size_t e = 0; e < instance.num_rounds; e++) {
        // std::vector<field::GF2E> Tau_e(1 << bit_len);
        // Tau_e[0] = field::GF2E(1);
        // std::vector<field::GF2E> AX_e(1 << bit_len);
        // Rho[e].resize(1 << bit_len);
        // Rho[e][0] = field::GF2E(1);

        Tau[e].resize(1 << bit_len);
        Rho[e].resize(1 << bit_len);
        Tau[e][0] = 1;
        Rho[e][0] = 1;
        AX[e].resize(1 << bit_len);

        for (int k = 0; k < bit_len; k++) {
            int mask = 1 << k;
            for (int i = 0; i < mask; i++) {
                Tau[e][i + mask] = field::mul(Tau[e][i], tau[k][e].data);
                Tau[e][i] ^= Tau[e][i + mask];
            }
        }

        // for (int k = 0; k < bit_len; k++) {
        //     int mask = 1 << k;
        //     if (mask >= 8) {
        //         __m256i t_lo = _mm256_set1_epi32((uint32_t)(tau[k][e].data & 255) << 16);
        //         __m256i t_hi = _mm256_set1_epi32((256 + (uint32_t)(tau[k][e].data >> 8)) << 16);
        //         for (int i = 0; i < mask / 8; i++) {
        //             __m256i t_i = _mm256_loadu_si256((__m256i *)&(Tau_v[e][i * 8]));
        //             __m256i tlo_i  = _mm256_xor_si256(t_i, t_lo);
        //             __m256i thi_i  = _mm256_xor_si256(t_i, t_hi);
        //             tlo_i = _mm256_i32gather_epi32((const int *)F::mul_lut, tlo_i, 4);
        //             thi_i = _mm256_i32gather_epi32((const int *)F::mul_lut, thi_i, 4);
        //             __m256i t_j = _mm256_xor_si256(tlo_i, thi_i);
        //             _mm256_storeu_si256((__m256i *)&(Tau_v[e][i * 8]), _mm256_xor_si256(t_i, t_j));
        //             _mm256_storeu_si256((__m256i *)&(Tau_v[e][i * 8 + mask]), t_j);

        //             // if (k == bit_len - 1)
        //             // for (int j = 0; j < 8; j++) {
        //             //     if (Tau_e[i * 8 + j].data != Tau_v[e][i * 8 + j])
        //             //         printf("wtf0\n");
        //             //     if(Tau_e[i * 8 + j + mask].data != Tau_v[e][i * 8 + j + mask])
        //             //         printf("wtf1\n");
        //             // }
        //         }
        //     } else
        //         for (int i = 0; i < mask; i++) {
        //             Tau_v[e][i + mask] = (field::GF2E(Tau_v[e][i]) * tau[k][e]).data;
        //             Tau_v[e][i] ^= Tau_v[e][i + mask];
        //         }
        // }

        for (int i = 0; i < (1 << (bit_len - 1)); i++) {
            int j = (1 << (bit_len - 1)) + i;
            AX[e][i] = field::mul(Tau[e][i], AV[e][j]);
            AX[e][j] = field::mul(Tau[e][j], AV[e][i]);
        }

        // for (int i = 0; i < (1 << (bit_len - 1)); i += 8) {
        //     int j = (1 << (bit_len - 1)) + i;

        //     __m256i vi = _mm256_loadu_si256((__m256i *)&(AV_v[e][i]));
        //     __m256i ti = _mm256_loadu_si256((__m256i *)&(Tau_v[e][i]));
        //     __m256i vj = _mm256_loadu_si256((__m256i *)&(AV_v[e][j]));
        //     __m256i tj = _mm256_loadu_si256((__m256i *)&(Tau_v[e][j]));

        //     __m256i ti_lo = _mm256_slli_epi32(_mm256_and_si256(_mm256_set1_epi32(255), ti), 16);
        //     __m256i ti_hi = _mm256_or_si256(
        //             _mm256_slli_epi32(_mm256_and_si256(_mm256_set1_epi32(255 << 8), ti), 8),
        //             _mm256_set1_epi32(256 * 65536));

        //     ti_lo = _mm256_i32gather_epi32((const int *)F::mul_lut, _mm256_xor_si256(ti_lo, vj), 4);
        //     ti_hi = _mm256_i32gather_epi32((const int *)F::mul_lut, _mm256_xor_si256(ti_hi, vj), 4);

        //     __m256i tj_lo = _mm256_slli_epi32(_mm256_and_si256(_mm256_set1_epi32(255), tj), 16);
        //     __m256i tj_hi = _mm256_or_si256(
        //             _mm256_slli_epi32(_mm256_and_si256(_mm256_set1_epi32(255 << 8), tj), 8),
        //             _mm256_set1_epi32(256 * 65536));
        //     tj_lo = _mm256_i32gather_epi32((const int *)F::mul_lut, _mm256_xor_si256(tj_lo, vi), 4);
        //     tj_hi = _mm256_i32gather_epi32((const int *)F::mul_lut, _mm256_xor_si256(tj_hi, vi), 4);

        //     _mm256_storeu_si256((__m256i *)&(AX_v[e][i]), _mm256_xor_si256(ti_lo, ti_hi));
        //     _mm256_storeu_si256((__m256i *)&(AX_v[e][j]), _mm256_xor_si256(tj_lo, tj_hi));

        //     // for (int k; k < 8; k++) {
        //     //     if (AX_v[e][i + k] != AX_e[i + k].data)
        //     //         printf("wtf ax0\n");
        //     //     if (AX_v[e][j + k] != AX_e[j + k].data)
        //     //         printf("wtf ax1\n");
        //     // }
        // }

    }
    AG = Tau;

#ifdef TIMING
    tmp_time = timing_read(&ctx);
    printf("gkr phase 1 init time: %ld\n", tmp_time - start_time);
    start_time = timing_read(&ctx);

    uint64_t eval_time1 = 0;
    uint64_t eval_time2 = 0;
    uint64_t update_time1 = 0;
    uint64_t update_time2 = 0;
    uint64_t update_time3 = 0;
    uint64_t tt;
#endif

    // sumcheck phase 1 loop
    for (int i = 0; i < bit_len; i++) {
        coef1[i].resize(instance.num_rounds);
        std::vector<std::vector<field::GF2E>> coef_i_deltas;

#ifdef TIMING
        tt = timing_read(&ctx);
#endif
        // phase 1 evaluate
        for (size_t e = 0; e < instance.num_rounds; e++) {

            // field::GF2E tmp(0);
            for (int u = 0; u < (1 << (bit_len - i)); u += 2) {
                int one_u = u + 1;
                AV[e][one_u] ^= AV[e][u];
                AG[e][one_u] ^= AG[e][u];
                AX[e][one_u] ^= AX[e][u];
                // tmp += AV[e][u] * AV[e][u] * AX[e][one_u] + AV[e][u] * AG[e][one_u] + AG[e][u] * AV[e][one_u];

                uint16_t tmp0, tmp1;
                field::pair_mul(AX[e][u], AX[e][u],
                                AV[e][u], AV[e][one_u],
                                &tmp0, &tmp1);
                // field::pair_mul(AX[e][u], AX[e][u], AX[e][one_u],
                //                 AV[e][u], AV[e][one_u], AV[e][one_u],
                //                 &tmp0, &tmp1, &tmp2);
                field::pair_mul(tmp0 ^ AG[e][u], tmp1 ^ AG[e][one_u], AV[e][u], AV[e][one_u], &tmp0, &tmp1);
                // field::pair_mul(tmp0 ^ AG[e][u], tmp1 ^ AG[e][one_u], tmp2,
                //                 AV[e][u], AV[e][one_u], AV[e][one_u],
                //                 &tmp0, &tmp1, &tmp2);
                coef1[i][e][0] += field::GF2E(tmp0);
                coef1[i][e][1] += field::GF2E(tmp1);
                // coef1[i][e][2] += field::GF2E(tmp2);
                coef1[i][e][2] += field::GF2E(field::mul(AV[e][one_u], field::mul(AV[e][one_u], AX[e][one_u])));

                // coef1[i][e][0] += AV[e][u] * (AV[e][u] * AX[e][u] + AG[e][u]);
                // coef1[i][e][1] += AV[e][one_u] * (AV[e][one_u] * AX[e][u] + AG[e][one_u]);
                // coef1[i][e][2] += field::GF2E(field::mul(AV[e][one_u], field::mul(AV[e][one_u], AX[e][one_u])));
            }

            // for (int u = 0; u < (1 << (bit_len - i)); u += 2) {
            //     int one_u = u + 1;
            //     // AH[e][one_u] -= AH[e][u];   // all zero
            //     AV[e][one_u] -= AV[e][u];
            //     AG[e][one_u] -= AG[e][u];
            //     AX[e][one_u] -= AX[e][u];
            //     // if (AG_v[e][u] != AG[e][u].data)
            //     //     printf("wtf\n");
            //     // tmp += AV[e][u] * AV[e][u] * AX[e][one_u] + AV[e][u] * AG[e][one_u] + AG[e][u] * AV[e][one_u];
            //     coef1[i][e][0] += AV[e][u] * AV[e][u] * AX[e][u] + AG[e][u];
            //     coef1[i][e][1] += AV[e][one_u] * (AV[e][one_u] * AX[e][u] + AG[e][one_u]);
            //     coef1[i][e][2] += AV[e][one_u] * AV[e][one_u] * AX[e][one_u];
            // }

            fr[e][0] -= coef1[i][e][1] + coef1[i][e][2];
            // if (tmp != fr[e][0])
            //     printf("phase 1 wtffff %d\n", i);
        }

#ifdef TIMING
        eval_time1 += timing_read(&ctx) - tt;
        tt = timing_read(&ctx);
#endif

        for (size_t e = 0; e < instance.num_rounds; e++) {
            std::vector<field::GF2E> coef_ie_deltas(3);

            __m256i coef_ie_deltas_v[3];
            for (int k = 0; k < 3; k++)
                coef_ie_deltas_v[k] = _mm256_setzero_si256();

            for (size_t k = 0; k < instance.num_MPC_parties / 16; k++) {
                for (size_t p = 0; p < 16; p++) {
                    auto random_coef_share =
                        random_tapes.get_bytes(e, k * 16 + p,
                                instance.aes_params.key_size +
                                instance.aes_params.block_size * instance.aes_params.num_blocks +
                                instance.aes_params.num_sboxes +
                                3 * i * instance.lambda,
                                3 * instance.lambda);
                    coef_shares_v[e][i][0][k * 32 + p] = *(uint8_t *)(random_coef_share.data());
                    coef_shares_v[e][i][1][k * 32 + p] = *(uint8_t *)(random_coef_share.data() + 2);
                    coef_shares_v[e][i][2][k * 32 + p] = *(uint8_t *)(random_coef_share.data() + 4);
                    coef_shares_v[e][i][0][k * 32 + 16 + p] = *(uint8_t *)(random_coef_share.data() + 1);
                    coef_shares_v[e][i][1][k * 32 + 16 + p] = *(uint8_t *)(random_coef_share.data() + 3);
                    coef_shares_v[e][i][2][k * 32 + 16 + p] = *(uint8_t *)(random_coef_share.data() + 5);
                }
                __m256i t_coef0 = _mm256_loadu_si256((__m256i *)&(coef_shares_v[e][i][0][k * 32]));
                __m256i t_coef1 = _mm256_loadu_si256((__m256i *)&(coef_shares_v[e][i][1][k * 32]));
                __m256i t_coef2 = _mm256_loadu_si256((__m256i *)&(coef_shares_v[e][i][2][k * 32]));
                __m256i t_fr    = _mm256_loadu_si256((__m256i *)&(fr_shares_v[e][k * 32]));
                t_fr = _mm256_xor_si256(_mm256_xor_si256(t_coef1, t_coef2), t_fr);
                _mm256_storeu_si256((__m256i *)&(fr_shares_v[e][k * 32]), t_fr);
                coef_ie_deltas_v[0] = _mm256_xor_si256(coef_ie_deltas_v[0], t_coef0);
                coef_ie_deltas_v[1] = _mm256_xor_si256(coef_ie_deltas_v[1], t_coef1);
                coef_ie_deltas_v[2] = _mm256_xor_si256(coef_ie_deltas_v[2], t_coef2);
            }
            coef_ie_deltas[0] = coef1[i][e][0] - field::GF2E(_mm256_hxor_epu16(coef_ie_deltas_v[0]));
            coef_ie_deltas[1] = coef1[i][e][1] - field::GF2E(_mm256_hxor_epu16(coef_ie_deltas_v[1]));
            coef_ie_deltas[2] = coef1[i][e][2] - field::GF2E(_mm256_hxor_epu16(coef_ie_deltas_v[2]));
            coef_i_deltas.push_back(coef_ie_deltas);

            coef_shares_v[e][i][0][0] ^= (coef_ie_deltas[0].data & 255);
            coef_shares_v[e][i][1][0] ^= (coef_ie_deltas[1].data & 255);
            coef_shares_v[e][i][2][0] ^= (coef_ie_deltas[2].data & 255);
            coef_shares_v[e][i][0][16] ^= (coef_ie_deltas[0].data >> 8);
            coef_shares_v[e][i][1][16] ^= (coef_ie_deltas[1].data >> 8);
            coef_shares_v[e][i][2][16] ^= (coef_ie_deltas[2].data >> 8);

            fr_shares_v[e][0] ^= (coef_ie_deltas[1] + coef_ie_deltas[2]).data & 255;
            fr_shares_v[e][16] ^= (coef_ie_deltas[1] + coef_ie_deltas[2]).data >> 8;

            // for (size_t p = 0; p < instance.num_MPC_parties; p++) {
            //     auto random_coef_share =
            //         random_tapes.get_bytes(e, p,
            //                 instance.aes_params.key_size + instance.aes_params.num_sboxes + 3 * i * instance.lambda,
            //                 3 * instance.lambda);
            //     coef_shares[e][i][0][p].from_bytes(random_coef_share.data());
            //     coef_shares[e][i][1][p].from_bytes(random_coef_share.data() + instance.lambda);
            //     coef_shares[e][i][2][p].from_bytes(random_coef_share.data() + instance.lambda * 2);

            //     fr_shares[e][p] -= coef_shares[e][i][1][p] + coef_shares[e][i][2][p];
            //     coef_ie_deltas[0] += coef_shares[e][i][0][p];
            //     coef_ie_deltas[1] += coef_shares[e][i][1][p];
            //     coef_ie_deltas[2] += coef_shares[e][i][2][p];
            // }
            // coef_ie_deltas[0] = coef1[i][e][0] - coef_ie_deltas[0];
            // coef_ie_deltas[1] = coef1[i][e][1] - coef_ie_deltas[1];
            // coef_ie_deltas[2] = coef1[i][e][2] - coef_ie_deltas[2];
            // coef_i_deltas.push_back(coef_ie_deltas);
            // coef_shares[e][i][0][0] += coef_ie_deltas[0];
            // coef_shares[e][i][1][0] += coef_ie_deltas[1];
            // coef_shares[e][i][2][0] += coef_ie_deltas[2];
            // fr_shares[e][0] += coef_ie_deltas[1] + coef_ie_deltas[2];

            // if ((fr_shares[e][1].data & 255) != fr_shares_v[e][1])
            //     printf("wtf\n");
            // if ((fr_shares[e][1].data >> 8) != fr_shares_v[e][17])
            //     printf("wtf2\n");

        }
        coef1_deltas.push_back(coef_i_deltas);

#ifdef TIMING
        eval_time2 += timing_read(&ctx) - tt;
        tt = timing_read(&ctx);
#endif

        h_i =
            // phase_gkr_commitment(instance, salt, keypair.second, message, message_len,
            //                    party_seed_commitments, rep_key_deltas, rep_t_deltas,
            //                    rep_output_broadcasts);
            phase_sumcheck_commitment(instance, salt, h_i, coef1_deltas[i]);

        std::vector<field::GF2E> rho_i = phase_sumcheck_expand(instance, h_i);
        rho.push_back(rho_i);

        for (size_t e = 0; e < instance.num_rounds; e++) {
            //
            auto r = rho_i[e];
            auto r_raw = r.data;
            // __m256i r_lo    = _mm256_set1_epi32((uint32_t)(r.data & 255) << 16);
            // __m256i r_hi    = _mm256_set1_epi32((256 + (uint32_t)(r.data >> 8)) << 16);
            fr[e][0] = ((coef1[i][e][2] * r + coef1[i][e][1]) * r + fr[e][0]) * r + coef1[i][e][0];
            // for (size_t p = 0; p < instance.num_MPC_parties; p++) {
            //     fr_shares[e][p] = ((coef_shares[e][i][2][p] * r + coef_shares[e][i][1][p]) * r + fr_shares[e][p]) * r + coef_shares[e][i][0][p];
            // }

#ifdef TIMING
        tt = timing_read(&ctx);
#endif
            const __m256i table0 = _mm256_loadu_si256((__m256i *)&(field::GF2E::mul_lut[r_raw][0]));
            const __m256i table1 = _mm256_loadu_si256((__m256i *)&(field::GF2E::mul_lut[r_raw][8]));
            const __m256i table2 = _mm256_loadu_si256((__m256i *)&(field::GF2E::mul_lut[r_raw][16]));
            const __m256i table3 = _mm256_loadu_si256((__m256i *)&(field::GF2E::mul_lut[r_raw][24]));
            const __m256i mask1 = _mm256_set1_epi8(0x0f);
            const __m256i mask2 = _mm256_set1_epi8(0xf0);
            // printf("r = %04lx\n", r_raw);
            // printf("table0: ");
            // print(table0);
            for (size_t k = 0; k < instance.num_MPC_parties / 16; k++) {
                __m256i lo, hi, tmp;
                __m256i t_coef0 = _mm256_loadu_si256((__m256i *)&(coef_shares_v[e][i][0][k * 32]));
                __m256i t_coef1 = _mm256_loadu_si256((__m256i *)&(coef_shares_v[e][i][1][k * 32]));
                __m256i t_coef2 = _mm256_loadu_si256((__m256i *)&(coef_shares_v[e][i][2][k * 32]));
                __m256i t_fr    = _mm256_loadu_si256((__m256i *)&(fr_shares_v[e][k * 32]));

                // coef[2] * r
                //
                lo = _mm256_and_si256(t_coef2, mask1);  // a0, a2
                lo = _mm256_shuffle_epi8(table0, lo);   // a0lo, a2hi
                hi = _mm256_srli_epi64(_mm256_and_si256(t_coef2, mask2), 4);    // a1, a3
                hi = _mm256_shuffle_epi8(table1, hi);   // a1lo, a3hi
                tmp = _mm256_xor_si256(hi, lo);
                t_coef2 = _mm256_permute4x64_epi64(t_coef2, 2 | (3 << 2) | (1 << 6));   // a2,a3; a0,a1
                lo = _mm256_and_si256(t_coef2, mask1);  // a2, a0
                lo = _mm256_shuffle_epi8(table2, lo);   // a2lo, a0hi
                hi = _mm256_srli_epi64(_mm256_and_si256(t_coef2, mask2), 4);    // a3, a1
                hi = _mm256_shuffle_epi8(table3, hi);   // a3lo, a1hi
                t_coef2 = _mm256_xor_si256(tmp, _mm256_xor_si256(hi, lo));

                // + coef[1]
                t_coef2 = _mm256_xor_si256(t_coef1, t_coef2);
                // * r
                lo = _mm256_and_si256(t_coef2, mask1);  // a0, a2
                lo = _mm256_shuffle_epi8(table0, lo);   // a0lo, a2hi
                hi = _mm256_srli_epi64(_mm256_and_si256(t_coef2, mask2), 4);    // a1, a3
                hi = _mm256_shuffle_epi8(table1, hi);   // a1lo, a3hi
                tmp = _mm256_xor_si256(hi, lo);
                t_coef2 = _mm256_permute4x64_epi64(t_coef2, 2 | (3 << 2) | (1 << 6));   // a2,a3; a0,a1
                lo = _mm256_and_si256(t_coef2, mask1);  // a2, a0
                lo = _mm256_shuffle_epi8(table2, lo);   // a2lo, a0hi
                hi = _mm256_srli_epi64(_mm256_and_si256(t_coef2, mask2), 4);    // a3, a1
                hi = _mm256_shuffle_epi8(table3, hi);   // a3lo, a1hi
                t_coef2 = _mm256_xor_si256(tmp, _mm256_xor_si256(hi, lo));
                // + fr
                t_coef2 = _mm256_xor_si256(t_fr, t_coef2);
                // * r
                lo = _mm256_and_si256(t_coef2, mask1);  // a0, a2
                lo = _mm256_shuffle_epi8(table0, lo);   // a0lo, a2hi
                hi = _mm256_srli_epi64(_mm256_and_si256(t_coef2, mask2), 4);    // a1, a3
                hi = _mm256_shuffle_epi8(table1, hi);   // a1lo, a3hi
                tmp = _mm256_xor_si256(hi, lo);
                t_coef2 = _mm256_permute4x64_epi64(t_coef2, 2 | (3 << 2) | (1 << 6));   // a2,a3; a0,a1
                lo = _mm256_and_si256(t_coef2, mask1);  // a2, a0
                lo = _mm256_shuffle_epi8(table2, lo);   // a2lo, a0hi
                hi = _mm256_srli_epi64(_mm256_and_si256(t_coef2, mask2), 4);    // a3, a1
                hi = _mm256_shuffle_epi8(table3, hi);   // a3lo, a1hi
                t_coef2 = _mm256_xor_si256(tmp, _mm256_xor_si256(hi, lo));
                // + coef[0]
                t_coef2 = _mm256_xor_si256(t_coef0, t_coef2);

                _mm256_storeu_si256((__m256i *)&(fr_shares_v[e][k * 32]), t_coef2);

                // for (int p = 0; p < 16; p++) {
                //     if (fr_shares_v[e][k * 32 + p] != (fr_shares[e][k * 16 + p].data & 255))
                //         printf("wtf0\n");
                //     if (fr_shares_v[e][k * 32 + p + 16] != (fr_shares[e][k * 16 + p].data >> 8))
                //         printf("wtf1\n");
                // }
            }

            // for (size_t p = 0; p < instance.num_MPC_parties / 8; p++) {
            //     __m256i t_coef0 = _mm256_loadu_si256((__m256i *)&(coef_shares_v[e][i][0][p * 8]));
            //     __m256i t_coef1 = _mm256_loadu_si256((__m256i *)&(coef_shares_v[e][i][1][p * 8]));
            //     __m256i t_coef2 = _mm256_loadu_si256((__m256i *)&(coef_shares_v[e][i][2][p * 8]));
            //     __m256i t_fr    = _mm256_loadu_si256((__m256i *)&(fr_shares_v[e][p * 8]));

            //     __m256i rlo_c  = _mm256_xor_si256(t_coef2, r_lo);
            //     __m256i rhi_c  = _mm256_xor_si256(t_coef2, r_hi);
            //     rlo_c = _mm256_i32gather_epi32((const int *)F::mul_lut, rlo_c, 4);
            //     rhi_c = _mm256_i32gather_epi32((const int *)F::mul_lut, rhi_c, 4);
            //     t_coef1 = _mm256_xor_si256(t_coef1, _mm256_xor_si256(rlo_c, rhi_c));

            //     rlo_c  = _mm256_xor_si256(t_coef1, r_lo);
            //     rhi_c  = _mm256_xor_si256(t_coef1, r_hi);
            //     rlo_c = _mm256_i32gather_epi32((const int *)F::mul_lut, rlo_c, 4);
            //     rhi_c = _mm256_i32gather_epi32((const int *)F::mul_lut, rhi_c, 4);
            //     t_fr = _mm256_xor_si256(t_fr, _mm256_xor_si256(rlo_c, rhi_c));

            //     rlo_c  = _mm256_xor_si256(t_fr, r_lo);
            //     rhi_c  = _mm256_xor_si256(t_fr, r_hi);
            //     rlo_c = _mm256_i32gather_epi32((const int *)F::mul_lut, rlo_c, 4);
            //     rhi_c = _mm256_i32gather_epi32((const int *)F::mul_lut, rhi_c, 4);
            //     t_fr = _mm256_xor_si256(t_coef0, _mm256_xor_si256(rlo_c, rhi_c));
            //     _mm256_storeu_si256((__m256i *)&(fr_shares_v[e][p * 8]), t_fr);

            //     // for (int k = 0; k < 8; k++)
            //     //     if (fr_shares_v[e][p * 8 + k] != fr_shares[e][p * 8 + k].data)
            //     //         printf("wtf\n");
            // }

#ifdef TIMING
        update_time2 += timing_read(&ctx) - tt;
        tt = timing_read(&ctx);
#endif

            // phase 1 update
            int mask = 1 << (bit_len - i - 1);
            // if (mask < 16)
                for (int u = 0; u < mask; u++) {
                    int u0 = u << 1;
                    int u1 = u0 + 1;
                    // for (size_t p = 0; p < instance.num_MPC_parties; p++)
                    //     inputs_rho_share[e][u][p] = inputs_rho_share[e][u0][p] + (inputs_rho_share[e][u1][p] - inputs_rho_share[e][u0][p]) * r;
                    for (size_t k = 0; k < instance.num_MPC_parties / 16; k++) {
                        __m256i lo, hi, tmp;
                        __m256i t_u0 = _mm256_loadu_si256((__m256i *)&(inputs_rho_share_v[e][u0][k * 32]));
                        __m256i t_u1 = _mm256_loadu_si256((__m256i *)&(inputs_rho_share_v[e][u1][k * 32]));
                        t_u1 = _mm256_xor_si256(t_u0, t_u1);

                        // (t_u1 + t_u0) * r
                        lo = _mm256_and_si256(t_u1, mask1);  // a0, a2
                        lo = _mm256_shuffle_epi8(table0, lo);   // a0lo, a2hi
                        hi = _mm256_srli_epi64(_mm256_and_si256(t_u1, mask2), 4);    // a1, a3
                        hi = _mm256_shuffle_epi8(table1, hi);   // a1lo, a3hi
                        tmp = _mm256_xor_si256(hi, lo);
                        t_u1 = _mm256_permute4x64_epi64(t_u1, 2 | (3 << 2) | (1 << 6));   // a2,a3; a0,a1
                        lo = _mm256_and_si256(t_u1, mask1);  // a2, a0
                        lo = _mm256_shuffle_epi8(table2, lo);   // a2lo, a0hi
                        hi = _mm256_srli_epi64(_mm256_and_si256(t_u1, mask2), 4);    // a3, a1
                        hi = _mm256_shuffle_epi8(table3, hi);   // a3lo, a1hi
                        t_u1 = _mm256_xor_si256(tmp, _mm256_xor_si256(hi, lo));

                        // + t_u0
                        t_u1 = _mm256_xor_si256(t_u0, t_u1);
                        _mm256_storeu_si256((__m256i *)&(inputs_rho_share_v[e][u][k * 32]), t_u1);
                    }

                    field::scale_mul(AV[e][u1], AG[e][u1], AX[e][u1], r.data,
                            AV[e][u0], AG[e][u0], AX[e][u0],
                            &AV[e][u], &AG[e][u], &AX[e][u]);
                    // AV[e][u] = AV[e][u0] ^ AV[e][u1];
                    // AG[e][u] = AG[e][u0] ^ AG[e][u1];
                    // AX[e][u] = AX[e][u0] ^ field::mul(AX[e][u1], r.data);
                }
            // else
            //     for (int uu = 0; uu < mask/16; uu++) {
            //         for (int u = 0; u < 16; u++) {
            //             int u0 = (uu * 16 + u) << 1;
            //             int u1 = u0 + 1;
            //             // for (size_t p = 0; p < instance.num_MPC_parties; p++)
            //             //     inputs_rho_share[e][u][p] = inputs_rho_share[e][u0][p] + (inputs_rho_share[e][u1][p] - inputs_rho_share[e][u0][p]) * r;
            //             for (size_t k = 0; k < instance.num_MPC_parties / 16; k++) {
            //                 __m256i lo, hi, tmp;
            //                 __m256i t_u0 = _mm256_loadu_si256((__m256i *)&(inputs_rho_share_v[e][u0][k * 32]));
            //                 __m256i t_u1 = _mm256_loadu_si256((__m256i *)&(inputs_rho_share_v[e][u1][k * 32]));
            //                 t_u1 = _mm256_xor_si256(t_u0, t_u1);

            //                 // (t_u1 + t_u0) * r
            //                 lo = _mm256_and_si256(t_u1, mask1);  // a0, a2
            //                 lo = _mm256_shuffle_epi8(table0, lo);   // a0lo, a2hi
            //                 hi = _mm256_srli_epi64(_mm256_and_si256(t_u1, mask2), 4);    // a1, a3
            //                 hi = _mm256_shuffle_epi8(table1, hi);   // a1lo, a3hi
            //                 tmp = _mm256_xor_si256(hi, lo);
            //                 t_u1 = _mm256_permute4x64_epi64(t_u1, 2 | (3 << 2) | (1 << 6));   // a2,a3; a0,a1
            //                 lo = _mm256_and_si256(t_u1, mask1);  // a2, a0
            //                 lo = _mm256_shuffle_epi8(table2, lo);   // a2lo, a0hi
            //                 hi = _mm256_srli_epi64(_mm256_and_si256(t_u1, mask2), 4);    // a3, a1
            //                 hi = _mm256_shuffle_epi8(table3, hi);   // a3lo, a1hi
            //                 t_u1 = _mm256_xor_si256(tmp, _mm256_xor_si256(hi, lo));

            //                 // + t_u0
            //                 t_u1 = _mm256_xor_si256(t_u0, t_u1);
            //                 _mm256_storeu_si256((__m256i *)&(inputs_rho_share_v[e][uu * 16 + u][k * 32]), t_u1);
            //             }
            //         }

            //         __m256i lo, hi, tmp;
            //         __m256i s_u0, s_u1, v_u0, v_u1;
            //         // AV
            //         s_u0 = _mm256_loadu_si256((__m256i *)&(AV[e][uu * 16 * 2]));
            //         s_u1 = _mm256_loadu_si256((__m256i *)&(AV[e][uu * 16 * 2 + 16]));
            //         v_u0 = _mm256_packus_epi32(
            //                 _mm256_and_si256(s_u0, _mm256_set1_epi32(0xffff)),
            //                 _mm256_and_si256(s_u1, _mm256_set1_epi32(0xffff)));
            //         v_u0 = _mm256_permute4x64_epi64(v_u0, 0b11011000);
            //         v_u1 = _mm256_packus_epi32(
            //                 _mm256_srli_epi32(s_u0, 16),
            //                 _mm256_srli_epi32(s_u1, 16));
            //         v_u1 = _mm256_permute4x64_epi64(v_u1, 0b11011000);
            //         // to alt mapping
            //         lo = _mm256_packus_epi16(_mm256_and_si256(v_u1, _mm256_set1_epi16(0xff)), _mm256_set1_epi64x(0));
            //         lo = _mm256_permute4x64_epi64(lo, 0b11011000);
            //         hi = _mm256_packus_epi16(_mm256_set1_epi64x(0), _mm256_srli_epi16(v_u1, 8));
            //         hi = _mm256_permute4x64_epi64(hi, 0b11011000);
            //         v_u1 = _mm256_xor_si256(lo, hi);

            //         // u1 * r
            //         lo = _mm256_and_si256(v_u1, mask1);  // a0, a2
            //         lo = _mm256_shuffle_epi8(table0, lo);   // a0lo, a2hi
            //         hi = _mm256_srli_epi64(_mm256_and_si256(v_u1, mask2), 4);    // a1, a3
            //         hi = _mm256_shuffle_epi8(table1, hi);   // a1lo, a3hi
            //         tmp = _mm256_xor_si256(hi, lo);
            //         v_u1 = _mm256_permute4x64_epi64(v_u1, 2 | (3 << 2) | (1 << 6));   // a2,a3; a0,a1
            //         lo = _mm256_and_si256(v_u1, mask1);  // a2, a0
            //         lo = _mm256_shuffle_epi8(table2, lo);   // a2lo, a0hi
            //         hi = _mm256_srli_epi64(_mm256_and_si256(v_u1, mask2), 4);    // a3, a1
            //         hi = _mm256_shuffle_epi8(table3, hi);   // a3lo, a1hi
            //         v_u1 = _mm256_xor_si256(tmp, _mm256_xor_si256(hi, lo));

            //         // to standard mapping
            //         v_u1 = _mm256_unpacklo_epi8(
            //                 _mm256_permute4x64_epi64(v_u1, 0b11011000),
            //                 _mm256_permute4x64_epi64(v_u1, 0b01110010));

            //         v_u1 = _mm256_xor_si256(v_u0, v_u1);

            //         _mm256_storeu_si256((__m256i *)&(AV[e][uu * 16]), v_u1);

            //         // AG
            //         s_u0 = _mm256_loadu_si256((__m256i *)&(AG[e][uu * 16 * 2]));
            //         s_u1 = _mm256_loadu_si256((__m256i *)&(AG[e][uu * 16 * 2 + 16]));
            //         v_u0 = _mm256_packus_epi32(
            //                 _mm256_and_si256(s_u0, _mm256_set1_epi32(0xffff)),
            //                 _mm256_and_si256(s_u1, _mm256_set1_epi32(0xffff)));
            //         v_u0 = _mm256_permute4x64_epi64(v_u0, 0b11011000);
            //         v_u1 = _mm256_packus_epi32(
            //                 _mm256_srli_epi32(s_u0, 16),
            //                 _mm256_srli_epi32(s_u1, 16));
            //         v_u1 = _mm256_permute4x64_epi64(v_u1, 0b11011000);
            //         // to alt mapping
            //         lo = _mm256_packus_epi16(_mm256_and_si256(v_u1, _mm256_set1_epi16(0xff)), _mm256_set1_epi64x(0));
            //         lo = _mm256_permute4x64_epi64(lo, 0b11011000);
            //         hi = _mm256_packus_epi16(_mm256_set1_epi64x(0), _mm256_srli_epi16(v_u1, 8));
            //         hi = _mm256_permute4x64_epi64(hi, 0b11011000);
            //         v_u1 = _mm256_xor_si256(lo, hi);

            //         // u1 * r
            //         lo = _mm256_and_si256(v_u1, mask1);  // a0, a2
            //         lo = _mm256_shuffle_epi8(table0, lo);   // a0lo, a2hi
            //         hi = _mm256_srli_epi64(_mm256_and_si256(v_u1, mask2), 4);    // a1, a3
            //         hi = _mm256_shuffle_epi8(table1, hi);   // a1lo, a3hi
            //         tmp = _mm256_xor_si256(hi, lo);
            //         v_u1 = _mm256_permute4x64_epi64(v_u1, 2 | (3 << 2) | (1 << 6));   // a2,a3; a0,a1
            //         lo = _mm256_and_si256(v_u1, mask1);  // a2, a0
            //         lo = _mm256_shuffle_epi8(table2, lo);   // a2lo, a0hi
            //         hi = _mm256_srli_epi64(_mm256_and_si256(v_u1, mask2), 4);    // a3, a1
            //         hi = _mm256_shuffle_epi8(table3, hi);   // a3lo, a1hi
            //         v_u1 = _mm256_xor_si256(tmp, _mm256_xor_si256(hi, lo));

            //         // to standard mapping
            //         v_u1 = _mm256_unpacklo_epi8(
            //                 _mm256_permute4x64_epi64(v_u1, 0b11011000),
            //                 _mm256_permute4x64_epi64(v_u1, 0b01110010));

            //         v_u1 = _mm256_xor_si256(v_u0, v_u1);

            //         _mm256_storeu_si256((__m256i *)&(AG[e][uu * 16]), v_u1);

            //         // AX
            //         s_u0 = _mm256_loadu_si256((__m256i *)&(AX[e][uu * 16 * 2]));
            //         s_u1 = _mm256_loadu_si256((__m256i *)&(AX[e][uu * 16 * 2 + 16]));
            //         v_u0 = _mm256_packus_epi32(
            //                 _mm256_and_si256(s_u0, _mm256_set1_epi32(0xffff)),
            //                 _mm256_and_si256(s_u1, _mm256_set1_epi32(0xffff)));
            //         v_u0 = _mm256_permute4x64_epi64(v_u0, 0b11011000);
            //         v_u1 = _mm256_packus_epi32(
            //                 _mm256_srli_epi32(s_u0, 16),
            //                 _mm256_srli_epi32(s_u1, 16));
            //         v_u1 = _mm256_permute4x64_epi64(v_u1, 0b11011000);
            //         // to alt mapping
            //         lo = _mm256_packus_epi16(_mm256_and_si256(v_u1, _mm256_set1_epi16(0xff)), _mm256_set1_epi64x(0));
            //         lo = _mm256_permute4x64_epi64(lo, 0b11011000);
            //         hi = _mm256_packus_epi16(_mm256_set1_epi64x(0), _mm256_srli_epi16(v_u1, 8));
            //         hi = _mm256_permute4x64_epi64(hi, 0b11011000);
            //         v_u1 = _mm256_xor_si256(lo, hi);

            //         // u1 * r
            //         lo = _mm256_and_si256(v_u1, mask1);  // a0, a2
            //         lo = _mm256_shuffle_epi8(table0, lo);   // a0lo, a2hi
            //         hi = _mm256_srli_epi64(_mm256_and_si256(v_u1, mask2), 4);    // a1, a3
            //         hi = _mm256_shuffle_epi8(table1, hi);   // a1lo, a3hi
            //         tmp = _mm256_xor_si256(hi, lo);
            //         v_u1 = _mm256_permute4x64_epi64(v_u1, 2 | (3 << 2) | (1 << 6));   // a2,a3; a0,a1
            //         lo = _mm256_and_si256(v_u1, mask1);  // a2, a0
            //         lo = _mm256_shuffle_epi8(table2, lo);   // a2lo, a0hi
            //         hi = _mm256_srli_epi64(_mm256_and_si256(v_u1, mask2), 4);    // a3, a1
            //         hi = _mm256_shuffle_epi8(table3, hi);   // a3lo, a1hi
            //         v_u1 = _mm256_xor_si256(tmp, _mm256_xor_si256(hi, lo));

            //         // to standard mapping
            //         v_u1 = _mm256_unpacklo_epi8(
            //                 _mm256_permute4x64_epi64(v_u1, 0b11011000),
            //                 _mm256_permute4x64_epi64(v_u1, 0b01110010));

            //         v_u1 = _mm256_xor_si256(v_u0, v_u1);

            //         _mm256_storeu_si256((__m256i *)&(AX[e][uu * 16]), v_u1);

            //         // for (int u = 0; u < 16; u++) {
            //         //     int u0 = (uu * 16 + u) << 1;
            //         //     int u1 = u0 + 1;
            //         //     AV[e][uu * 16 + u] = AV[e][u0] ^ field::mul(AV[e][u1], r.data);
            //         //     AG[e][uu * 16 + u] = AG[e][u0] ^ field::mul(AG[e][u1], r.data);
            //         //     AX[e][uu * 16 + u] = AX[e][u0] ^ field::mul(AX[e][u1], r.data);
            //         // }
            //     }

            // if (mask < 8)
            //     for (int u = 0; u < mask; u++) {
            //         int u0 = u << 1;
            //         int u1 = u0 + 1;
            //         for (size_t p = 0; p < instance.num_MPC_parties / 8; p++) {
            //             __m256i t_u0 = _mm256_loadu_si256((__m256i *)&(inputs_rho_share_v[e][u0][p * 8]));
            //             __m256i t_u1 = _mm256_loadu_si256((__m256i *)&(inputs_rho_share_v[e][u1][p * 8]));
            //             t_u1 = _mm256_xor_si256(t_u1, t_u0);
            //             __m256i rlo_c  = _mm256_xor_si256(t_u1, r_lo);
            //             __m256i rhi_c  = _mm256_xor_si256(t_u1, r_hi);
            //             rlo_c = _mm256_i32gather_epi32((const int *)F::mul_lut, rlo_c, 4);
            //             rhi_c = _mm256_i32gather_epi32((const int *)F::mul_lut, rhi_c, 4);
            //             t_u1 = _mm256_xor_si256(t_u0, _mm256_xor_si256(rlo_c, rhi_c));
            //             _mm256_storeu_si256((__m256i *)&(inputs_rho_share_v[e][u][p * 8]), t_u1);
            //         }
            //         AV[e][u] = AV[e][u0] + AV[e][u1] * r;
            //         AG[e][u] = AG[e][u0] + AG[e][u1] * r;
            //         AX[e][u] = AX[e][u0] + AX[e][u1] * r;
            //     }
            // else
            //     for (int uu = 0; uu < mask; uu += 8) {
            //         for (int u = uu; u < uu + 8; u++) {
            //             int u0 = u << 1;
            //             int u1 = u0 + 1;
            //             for (size_t p = 0; p < instance.num_MPC_parties / 8; p++) {
            //                 __m256i t_u0 = _mm256_loadu_si256((__m256i *)&(inputs_rho_share_v[e][u0][p * 8]));
            //                 __m256i t_u1 = _mm256_loadu_si256((__m256i *)&(inputs_rho_share_v[e][u1][p * 8]));
            //                 t_u1 = _mm256_xor_si256(t_u1, t_u0);
            //                 __m256i rlo_c  = _mm256_xor_si256(t_u1, r_lo);
            //                 __m256i rhi_c  = _mm256_xor_si256(t_u1, r_hi);
            //                 rlo_c = _mm256_i32gather_epi32((const int *)F::mul_lut, rlo_c, 4);
            //                 rhi_c = _mm256_i32gather_epi32((const int *)F::mul_lut, rhi_c, 4);
            //                 t_u1 = _mm256_xor_si256(t_u0, _mm256_xor_si256(rlo_c, rhi_c));
            //                 _mm256_storeu_si256((__m256i *)&(inputs_rho_share_v[e][u][p * 8]), t_u1);
            //             }
            //         }

            //         // AV_v[e][uu + u] = AV_v[e][(uu + u) << 1] ^ (field::GF2E(AV_v[e][((uu + u) << 1) + 1]) * r).data;
            //         __m256i t_av = _mm256_loadu_si256((__m256i *)&(AV_v[e][uu << 1]));
            //         t_av = _mm256_permutevar8x32_epi32(t_av, _mm256_set_epi32(7, 5, 3, 1, 6, 4, 2, 0));
            //         __m128i t_evn = _mm256_castsi256_si128(t_av);
            //         __m128i t_odd = _mm256_extracti128_si256(t_av, 1);
            //         t_av = _mm256_loadu_si256((__m256i *)&(AV_v[e][(uu << 1) + 8]));
            //         t_av = _mm256_permutevar8x32_epi32(t_av, _mm256_set_epi32(7, 5, 3, 1, 6, 4, 2, 0));

            //         __m256i av_evn = _mm256_set_m128i(_mm256_castsi256_si128(t_av), t_evn);
            //         __m256i av_odd = _mm256_set_m128i(_mm256_extracti128_si256(t_av, 1), t_odd);

            //         __m256i rlo_odd = _mm256_xor_si256(av_odd, r_lo);
            //         __m256i rhi_odd = _mm256_xor_si256(av_odd, r_hi);

            //         rlo_odd = _mm256_i32gather_epi32((const int *)F::mul_lut, rlo_odd, 4);
            //         rhi_odd = _mm256_i32gather_epi32((const int *)F::mul_lut, rhi_odd, 4);
            //         t_av = _mm256_xor_si256(av_evn, _mm256_xor_si256(rlo_odd, rhi_odd));
            //         _mm256_storeu_si256((__m256i *)&(AV_v[e][uu]), t_av);

            //         // AG_v[e][uu + u] = AG_v[e][(uu + u) << 1] ^ (field::GF2E(AG_v[e][((uu + u) << 1) + 1]) * r).data;
            //         t_av = _mm256_loadu_si256((__m256i *)&(AG_v[e][uu << 1]));
            //         t_av = _mm256_permutevar8x32_epi32(t_av, _mm256_set_epi32(7, 5, 3, 1, 6, 4, 2, 0));
            //         t_evn = _mm256_castsi256_si128(t_av);
            //         t_odd = _mm256_extracti128_si256(t_av, 1);
            //         t_av = _mm256_loadu_si256((__m256i *)&(AG_v[e][(uu << 1) + 8]));
            //         t_av = _mm256_permutevar8x32_epi32(t_av, _mm256_set_epi32(7, 5, 3, 1, 6, 4, 2, 0));

            //         av_evn = _mm256_set_m128i(_mm256_castsi256_si128(t_av), t_evn);
            //         av_odd = _mm256_set_m128i(_mm256_extracti128_si256(t_av, 1), t_odd);

            //         rlo_odd = _mm256_xor_si256(av_odd, r_lo);
            //         rhi_odd = _mm256_xor_si256(av_odd, r_hi);

            //         rlo_odd = _mm256_i32gather_epi32((const int *)F::mul_lut, rlo_odd, 4);
            //         rhi_odd = _mm256_i32gather_epi32((const int *)F::mul_lut, rhi_odd, 4);
            //         t_av = _mm256_xor_si256(av_evn, _mm256_xor_si256(rlo_odd, rhi_odd));
            //         _mm256_storeu_si256((__m256i *)&(AG_v[e][uu]), t_av);

            //         // AX_v[e][uu + u] = AX_v[e][(uu + u) << 1] ^ (field::GF2E(AX_v[e][((uu + u) << 1) + 1]) * r).data;
            //         t_av = _mm256_loadu_si256((__m256i *)&(AX_v[e][uu << 1]));
            //         t_av = _mm256_permutevar8x32_epi32(t_av, _mm256_set_epi32(7, 5, 3, 1, 6, 4, 2, 0));
            //         t_evn = _mm256_castsi256_si128(t_av);
            //         t_odd = _mm256_extracti128_si256(t_av, 1);
            //         t_av = _mm256_loadu_si256((__m256i *)&(AX_v[e][(uu << 1) + 8]));
            //         t_av = _mm256_permutevar8x32_epi32(t_av, _mm256_set_epi32(7, 5, 3, 1, 6, 4, 2, 0));

            //         av_evn = _mm256_set_m128i(_mm256_castsi256_si128(t_av), t_evn);
            //         av_odd = _mm256_set_m128i(_mm256_extracti128_si256(t_av, 1), t_odd);

            //         rlo_odd = _mm256_xor_si256(av_odd, r_lo);
            //         rhi_odd = _mm256_xor_si256(av_odd, r_hi);

            //         rlo_odd = _mm256_i32gather_epi32((const int *)F::mul_lut, rlo_odd, 4);
            //         rhi_odd = _mm256_i32gather_epi32((const int *)F::mul_lut, rhi_odd, 4);
            //         t_av = _mm256_xor_si256(av_evn, _mm256_xor_si256(rlo_odd, rhi_odd));
            //         _mm256_storeu_si256((__m256i *)&(AX_v[e][uu]), t_av);

            //         // for (int u = 0; u < 8; u++) {
            //         //     // AV_v[e][uu + u] = AV_v[e][(uu + u) << 1] ^ (field::GF2E(AV_v[e][((uu + u) << 1) + 1]) * r).data;
            //         //     // AG_v[e][uu + u] = AG_v[e][(uu + u) << 1] ^ (field::GF2E(AG_v[e][((uu + u) << 1) + 1]) * r).data;
            //         //     // AX_v[e][uu + u] = AX_v[e][(uu + u) << 1] ^ (field::GF2E(AX_v[e][((uu + u) << 1) + 1]) * r).data;
            //         // }
            //     }

#ifdef TIMING
        update_time1 += timing_read(&ctx) - tt;
        tt = timing_read(&ctx);
#endif

            // build table Rho
            mask = 1 << i;
            for (int k = 0; k < mask; k++) {
                Rho[e][k + mask] = field::mul(Rho[e][k], r.data);
                Rho[e][k] ^= Rho[e][k + mask];
            }
#ifdef TIMING
        update_time3 += timing_read(&ctx) - tt;
#endif

            // if (mask >= 8) {
            //     for (int k = 0; k < mask / 8; k++) {
            //         __m256i r_i = _mm256_loadu_si256((__m256i *)&(Rho_v[e][k * 8]));
            //         __m256i rlo_i  = _mm256_xor_si256(r_i, r_lo);
            //         __m256i rhi_i  = _mm256_xor_si256(r_i, r_hi);
            //         rlo_i = _mm256_i32gather_epi32((const int *)F::mul_lut, rlo_i, 4);
            //         rhi_i = _mm256_i32gather_epi32((const int *)F::mul_lut, rhi_i, 4);
            //         __m256i r_j = _mm256_xor_si256(rlo_i, rhi_i);
            //         _mm256_storeu_si256((__m256i *)&(Rho_v[e][k * 8]), _mm256_xor_si256(r_i, r_j));
            //         _mm256_storeu_si256((__m256i *)&(Rho_v[e][k * 8 + mask]), r_j);

            //     }
            // } else
            //     for (int k = 0; k < mask; k++) {
            //         Rho_v[e][k + mask] = (field::GF2E(Rho_v[e][k]) * r).data;
            //         Rho_v[e][k] ^= Rho_v[e][k + mask];
            //     }
        }
    }

    // for (size_t e = 0; e < instance.num_rounds; e++)
    //     for (int i = 0; i < (1 << bit_len); i++)
    //         if (Rho[e][i].data != Rho_v[e][i])
    //             printf("wtf rho\n");

#ifdef TIMING
    tmp_time = timing_read(&ctx);
    printf("gkr phase 1 loop time: %ld\n", tmp_time - start_time);
    printf("gkr phase 1 eval1 time: %ld\n", eval_time1);
    printf("gkr phase 1 eval2 time: %ld\n", eval_time2);
    printf("gkr phase 1 update1 time: %ld\n", update_time1);
    printf("gkr phase 1 update2 time: %ld\n", update_time2);
    printf("gkr phase 1 update3 time: %ld\n", update_time3);
    start_time = timing_read(&ctx);
#endif

    // sumcheck phase 2 init
    for (size_t e = 0; e < instance.num_rounds; e++) {
        Vr[e][0] = AV[e][0];
        auto Vr_sq = field::mul(AV[e][0], AV[e][0]);

        // __m256i vr_lo = _mm256_set1_epi32((uint32_t)(Vr[e][0].data & 255) << 16);
        // __m256i vr_hi = _mm256_set1_epi32((256 + (uint32_t)(Vr[e][0].data >> 8)) << 16);

        AG[e].resize(1 << bit_len);
        AX[e].resize(1 << bit_len);

        for (int i = 0; i < (1 << bit_len); i++) {
            AG[e][i] = 0;
            AX[e][i] = 0;
        }

        for (int i = 0; i < (1 << (bit_len - 1)); i++) {
            int j = (1 << (bit_len - 1)) + i;

            auto tmp = field::mul(Tau[e][i], Rho[e][i]);
            AG[e][j] = field::mul(tmp, AV[e][0]);
            AX[e][j] = field::mul(tmp, Vr_sq);

            tmp = field::mul(Tau[e][j], Rho[e][j]);
            AG[e][i] = field::mul(tmp, AV[e][0]);
            AX[e][i] = field::mul(tmp, Vr_sq);
        }

        // for (int i = 0; i < (1 << (bit_len - 1)); i += 8) {
        //     int j = (1 << (bit_len - 1)) + i;

        //     __m256i ti = _mm256_loadu_si256((__m256i *)&(Tau_v[e][i]));
        //     __m256i tj = _mm256_loadu_si256((__m256i *)&(Tau_v[e][j]));
        //     __m256i ri = _mm256_loadu_si256((__m256i *)&(Rho_v[e][i]));
        //     __m256i rj = _mm256_loadu_si256((__m256i *)&(Rho_v[e][j]));

        //     __m256i ti_lo = _mm256_slli_epi32(_mm256_and_si256(_mm256_set1_epi32(255), ti), 16);
        //     __m256i ti_hi = _mm256_or_si256(
        //             _mm256_slli_epi32(_mm256_and_si256(_mm256_set1_epi32(255 << 8), ti), 8),
        //             _mm256_set1_epi32(256 * 65536));

        //     ti_lo = _mm256_i32gather_epi32((const int *)F::mul_lut, _mm256_xor_si256(ti_lo, ri), 4);
        //     ti_hi = _mm256_i32gather_epi32((const int *)F::mul_lut, _mm256_xor_si256(ti_hi, ri), 4);

        //     __m256i tj_lo = _mm256_slli_epi32(_mm256_and_si256(_mm256_set1_epi32(255), tj), 16);
        //     __m256i tj_hi = _mm256_or_si256(
        //             _mm256_slli_epi32(_mm256_and_si256(_mm256_set1_epi32(255 << 8), tj), 8),
        //             _mm256_set1_epi32(256 * 65536));
        //     tj_lo = _mm256_i32gather_epi32((const int *)F::mul_lut, _mm256_xor_si256(tj_lo, rj), 4);
        //     tj_hi = _mm256_i32gather_epi32((const int *)F::mul_lut, _mm256_xor_si256(tj_hi, rj), 4);

        //     __m256i tmpi = _mm256_xor_si256(tj_lo, tj_hi);
        //     __m256i tmpj = _mm256_xor_si256(ti_lo, ti_hi);

        //     __m256i lo  = _mm256_xor_si256(tmpi, vr_lo);
        //     __m256i hi  = _mm256_xor_si256(tmpi, vr_hi);
        //     lo = _mm256_i32gather_epi32((const int *)F::mul_lut, lo, 4);
        //     hi = _mm256_i32gather_epi32((const int *)F::mul_lut, hi, 4);
        //     tmpi = _mm256_xor_si256(lo, hi);
        //     _mm256_storeu_si256((__m256i *)&(AG_v[e][i]), tmpi);

        //     lo  = _mm256_xor_si256(tmpi, vr_lo);
        //     hi  = _mm256_xor_si256(tmpi, vr_hi);
        //     lo = _mm256_i32gather_epi32((const int *)F::mul_lut, lo, 4);
        //     hi = _mm256_i32gather_epi32((const int *)F::mul_lut, hi, 4);
        //     tmpi = _mm256_xor_si256(lo, hi);
        //     _mm256_storeu_si256((__m256i *)&(AX_v[e][i]), tmpi);

        //     lo  = _mm256_xor_si256(tmpj, vr_lo);
        //     hi  = _mm256_xor_si256(tmpj, vr_hi);
        //     lo = _mm256_i32gather_epi32((const int *)F::mul_lut, lo, 4);
        //     hi = _mm256_i32gather_epi32((const int *)F::mul_lut, hi, 4);
        //     tmpj = _mm256_xor_si256(lo, hi);
        //     _mm256_storeu_si256((__m256i *)&(AG_v[e][j]), tmpj);

        //     lo  = _mm256_xor_si256(tmpj, vr_lo);
        //     hi  = _mm256_xor_si256(tmpj, vr_hi);
        //     lo = _mm256_i32gather_epi32((const int *)F::mul_lut, lo, 4);
        //     hi = _mm256_i32gather_epi32((const int *)F::mul_lut, hi, 4);
        //     tmpj = _mm256_xor_si256(lo, hi);
        //     _mm256_storeu_si256((__m256i *)&(AX_v[e][j]), tmpj);

        //     // for (int k = 0; k < 8; k++) {
        //     //     if (AX_v[e][i + k] != AX[e][i + k].data)
        //     //         printf("wtf ax0\n");
        //     //     if (AX_v[e][j + k] != AX[e][j + k].data)
        //     //         printf("wtf ax1\n");
        //     //     if (AG_v[e][i + k] != AG[e][i + k].data)
        //     //         printf("wtf ag0\n");
        //     //     if (AG_v[e][j + k] != AG[e][j + k].data)
        //     //         printf("wtf ag1\n");
        //     // }
        // }

    }
    // AV = inputs;
    AV = inputs;

#ifdef TIMING
    tmp_time = timing_read(&ctx);
    printf("gkr phase 2 init time: %ld\n", tmp_time - start_time);
    start_time = timing_read(&ctx);
#endif

    // sumcheck phase 2
    for (int i = 0; i < bit_len; i++) {
        // phase 2 evaluate
        coef2[i].resize(instance.num_rounds);
        std::vector<std::vector<field::GF2E>> coef_i_deltas;
        for (size_t e = 0; e < instance.num_rounds; e++) {
            // field::GF2E tmp(0);
            for (int u = 0; u < (1 << (bit_len - i)); u += 2) {
                int one_u = u + 1;
                AV[e][one_u] ^= AV[e][u];
                AG[e][one_u] ^= AG[e][u];
                AX[e][one_u] ^= AX[e][u];

                // tmp += AV[e][one_u] * AX[e][u] + AV[e][u] * AX[e][one_u] + AG[e][one_u];
                // coef2[i][e][0] += field::mul(AV[e][u], AX[e][u]) ^ AG[e][u];
                // coef2[i][e][1] += field::mul(AV[e][one_u], AX[e][one_u]);
                uint16_t tmp0, tmp1;
                field::pair_mul(AV[e][u], AV[e][one_u], AX[e][u], AX[e][one_u], &tmp0, &tmp1);
                coef2[i][e][0] += field::GF2E(tmp0 ^ AG[e][u]);
                coef2[i][e][1] += field::GF2E(tmp1);

            }
            // for (int u = 0; u < (1 << (bit_len - i)); u += 2) {
            //     int one_u = u + 1;
            //     AV_v[e][one_u] ^= AV_v[e][u];
            //     AG_v[e][one_u] ^= AG_v[e][u];
            //     AX_v[e][one_u] ^= AX_v[e][u];

            //     // tmp += AV[e][one_u] * AX[e][u] + AV[e][u] * AX[e][one_u] + AG[e][one_u];
            //     coef2[i][e][0] += field::GF2E(AV_v[e][u]) * field::GF2E(AX_v[e][u]) + field::GF2E(AG_v[e][u]);
            //     coef2[i][e][1] += field::GF2E(AV_v[e][one_u]) * field::GF2E(AX_v[e][one_u]);

            // }
            fr[e][0] -= coef2[i][e][1];
            // if (tmp != fr[e][0])
            //     printf("wtffffffffffff %d\n", i);
        // }

        // for (size_t e = 0; e < instance.num_rounds; e++) {
            std::vector<field::GF2E> coef_ie_deltas(2);
            __m256i coef_ie_deltas_v[3];
            for (int k = 0; k < 3; k++)
                coef_ie_deltas_v[k] = _mm256_setzero_si256();

            for (size_t k = 0; k < instance.num_MPC_parties / 16; k++) {
                for (size_t p = 0; p < 16; p++) {
                    auto random_coef_share =
                        random_tapes.get_bytes(e, k * 16 + p,
                                instance.aes_params.key_size +
                                instance.aes_params.block_size * instance.aes_params.num_blocks +
                                instance.aes_params.num_sboxes +
                                3 * bit_len * instance.lambda +
                                2 * i * instance.lambda,
                                2 * instance.lambda);
                    coef_shares_v[e][i][3][k * 32 + p] = *(uint8_t *)(random_coef_share.data());
                    coef_shares_v[e][i][4][k * 32 + p] = *(uint8_t *)(random_coef_share.data() + 2);
                    coef_shares_v[e][i][3][k * 32 + 16 + p] = *(uint8_t *)(random_coef_share.data() + 1);
                    coef_shares_v[e][i][4][k * 32 + 16 + p] = *(uint8_t *)(random_coef_share.data() + 3);
                }
                __m256i t_coef0 = _mm256_loadu_si256((__m256i *)&(coef_shares_v[e][i][3][k * 32]));
                __m256i t_coef1 = _mm256_loadu_si256((__m256i *)&(coef_shares_v[e][i][4][k * 32]));
                __m256i t_fr    = _mm256_loadu_si256((__m256i *)&(fr_shares_v[e][k * 32]));
                t_fr = _mm256_xor_si256(t_coef1, t_fr);
                _mm256_storeu_si256((__m256i *)&(fr_shares_v[e][k * 32]), t_fr);
                coef_ie_deltas_v[0] = _mm256_xor_si256(coef_ie_deltas_v[0], t_coef0);
                coef_ie_deltas_v[1] = _mm256_xor_si256(coef_ie_deltas_v[1], t_coef1);
            }
            coef_ie_deltas[0] = coef2[i][e][0] - field::GF2E(_mm256_hxor_epu16(coef_ie_deltas_v[0]));
            coef_ie_deltas[1] = coef2[i][e][1] - field::GF2E(_mm256_hxor_epu16(coef_ie_deltas_v[1]));
            coef_i_deltas.push_back(coef_ie_deltas);

            coef_shares_v[e][i][3][0] ^= (coef_ie_deltas[0].data & 255);
            coef_shares_v[e][i][4][0] ^= (coef_ie_deltas[1].data & 255);
            coef_shares_v[e][i][3][16] ^= (coef_ie_deltas[0].data >> 8);
            coef_shares_v[e][i][4][16] ^= (coef_ie_deltas[1].data >> 8);

            fr_shares_v[e][0] ^= coef_ie_deltas[1].data & 255;
            fr_shares_v[e][16] ^= coef_ie_deltas[1].data >> 8;


            // for (size_t p = 0; p < instance.num_MPC_parties; p++) {
            //     auto random_coef_share =
            //         random_tapes.get_bytes(e, p,
            //                 instance.aes_params.key_size +
            //                 instance.aes_params.num_sboxes +
            //                 3 * bit_len * instance.lambda +
            //                 2 * i * instance.lambda,
            //                 2 * instance.lambda);
            //     coef_shares[e][i][3][p].from_bytes(random_coef_share.data());
            //     coef_shares[e][i][4][p].from_bytes(random_coef_share.data() + instance.lambda);

            //     fr_shares[e][p] -= coef_shares[e][i][4][p];
            //     coef_ie_deltas[0] += coef_shares[e][i][3][p];
            //     coef_ie_deltas[1] += coef_shares[e][i][4][p];
            // }
            // coef_ie_deltas[0] = coef2[i][e][0] - coef_ie_deltas[0];
            // coef_ie_deltas[1] = coef2[i][e][1] - coef_ie_deltas[1];
            // coef_i_deltas.push_back(coef_ie_deltas);
            // coef_shares[e][i][3][0] += coef_ie_deltas[0];
            // coef_shares[e][i][4][0] += coef_ie_deltas[1];
            // fr_shares[e][0] += coef_ie_deltas[1];
        }
        coef2_deltas.push_back(coef_i_deltas);

        h_i =
            // phase_gkr_commitment(instance, salt, keypair.second, message, message_len,
            //                    party_seed_commitments, rep_key_deltas, rep_t_deltas,
            //                    rep_output_broadcasts);
            phase_sumcheck_commitment(instance, salt, h_i, coef2_deltas[i]);

        std::vector<field::GF2E> sgm = phase_sumcheck_expand(instance, h_i);
        for (size_t e = 0; e < instance.num_rounds; e++) {
            auto r = sgm[e];
            auto r_raw = r.data;
            // __m256i r_lo = _mm256_set1_epi32((uint32_t)(r.data & 255) << 16);
            // __m256i r_hi = _mm256_set1_epi32((256 + (uint32_t)(r.data >> 8)) << 16);
            fr[e][0] = (coef2[i][e][1] * r + fr[e][0]) * r + coef2[i][e][0];
            // for (size_t p = 0; p < instance.num_MPC_parties; p++) {
            //     fr_shares[e][p] = (coef_shares[e][i][4][p] * r + fr_shares[e][p]) * r + coef_shares[e][i][3][p];
            // }
            __m256i table0 = _mm256_loadu_si256((__m256i *)&(field::GF2E::mul_lut[r_raw][0]));
            __m256i table1 = _mm256_loadu_si256((__m256i *)&(field::GF2E::mul_lut[r_raw][8]));
            __m256i table2 = _mm256_loadu_si256((__m256i *)&(field::GF2E::mul_lut[r_raw][16]));
            __m256i table3 = _mm256_loadu_si256((__m256i *)&(field::GF2E::mul_lut[r_raw][24]));
            __m256i mask1 = _mm256_set1_epi8(0x0f);
            __m256i mask2 = _mm256_set1_epi8(0xf0);
            for (size_t k = 0; k < instance.num_MPC_parties / 16; k++) {
                __m256i lo, hi, tmp;
                __m256i t_coef0 = _mm256_loadu_si256((__m256i *)&(coef_shares_v[e][i][3][k * 32]));
                __m256i t_coef1 = _mm256_loadu_si256((__m256i *)&(coef_shares_v[e][i][4][k * 32]));
                __m256i t_fr    = _mm256_loadu_si256((__m256i *)&(fr_shares_v[e][k * 32]));

                // coef1 * r
                lo = _mm256_and_si256(t_coef1, mask1);  // a0, a2
                lo = _mm256_shuffle_epi8(table0, lo);   // a0lo, a2hi
                hi = _mm256_srli_epi64(_mm256_and_si256(t_coef1, mask2), 4);    // a1, a3
                hi = _mm256_shuffle_epi8(table1, hi);   // a1lo, a3hi
                tmp = _mm256_xor_si256(hi, lo);
                t_coef1 = _mm256_permute4x64_epi64(t_coef1, 2 | (3 << 2) | (1 << 6));   // a2,a3; a0,a1
                lo = _mm256_and_si256(t_coef1, mask1);  // a2, a0
                lo = _mm256_shuffle_epi8(table2, lo);   // a2lo, a0hi
                hi = _mm256_srli_epi64(_mm256_and_si256(t_coef1, mask2), 4);    // a3, a1
                hi = _mm256_shuffle_epi8(table3, hi);   // a3lo, a1hi
                t_coef1 = _mm256_xor_si256(tmp, _mm256_xor_si256(hi, lo));
                // + fr
                t_coef1 = _mm256_xor_si256(t_fr, t_coef1);
                // * r
                lo = _mm256_and_si256(t_coef1, mask1);  // a0, a2
                lo = _mm256_shuffle_epi8(table0, lo);   // a0lo, a2hi
                hi = _mm256_srli_epi64(_mm256_and_si256(t_coef1, mask2), 4);    // a1, a3
                hi = _mm256_shuffle_epi8(table1, hi);   // a1lo, a3hi
                tmp = _mm256_xor_si256(hi, lo);
                t_coef1 = _mm256_permute4x64_epi64(t_coef1, 2 | (3 << 2) | (1 << 6));   // a2,a3; a0,a1
                lo = _mm256_and_si256(t_coef1, mask1);  // a2, a0
                lo = _mm256_shuffle_epi8(table2, lo);   // a2lo, a0hi
                hi = _mm256_srli_epi64(_mm256_and_si256(t_coef1, mask2), 4);    // a3, a1
                hi = _mm256_shuffle_epi8(table3, hi);   // a3lo, a1hi
                t_coef1 = _mm256_xor_si256(tmp, _mm256_xor_si256(hi, lo));
                // + coef[0]
                t_coef1 = _mm256_xor_si256(t_coef0, t_coef1);

                _mm256_storeu_si256((__m256i *)&(fr_shares_v[e][k * 32]), t_coef1);

                // for (int p = 0; p < 16; p++) {
                //     if (fr_shares_v[e][k * 32 + p] != (fr_shares[e][k * 16 + p].data & 255))
                //         printf("wtf0\n");
                //     if (fr_shares_v[e][k * 32 + p + 16] != (fr_shares[e][k * 16 + p].data >> 8))
                //         printf("wtf1\n");
                // }
            }


            int mask = 1 << (bit_len - i - 1);
            // if (mask < 65536)
                for (int u = 0; u < mask; u++) {
                    int u0 = u << 1;
                    int u1 = u0 + 1;
                    // for (size_t p = 0; p < instance.num_MPC_parties; p++)
                    //     inputs_sgm_share[e][u][p] = inputs_sgm_share[e][u0][p] + (inputs_sgm_share[e][u1][p] - inputs_sgm_share[e][u0][p]) * r;
                    for (size_t k = 0; k < instance.num_MPC_parties / 16; k++) {
                        __m256i lo, hi, tmp;
                        __m256i t_u0 = _mm256_loadu_si256((__m256i *)&(inputs_sgm_share_v[e][u0][k * 32]));
                        __m256i t_u1 = _mm256_loadu_si256((__m256i *)&(inputs_sgm_share_v[e][u1][k * 32]));
                        t_u1 = _mm256_xor_si256(t_u0, t_u1);

                        // (t_u1 + t_u0) * r
                        lo = _mm256_and_si256(t_u1, mask1);  // a0, a2
                        lo = _mm256_shuffle_epi8(table0, lo);   // a0lo, a2hi
                        hi = _mm256_srli_epi64(_mm256_and_si256(t_u1, mask2), 4);    // a1, a3
                        hi = _mm256_shuffle_epi8(table1, hi);   // a1lo, a3hi
                        tmp = _mm256_xor_si256(hi, lo);
                        t_u1 = _mm256_permute4x64_epi64(t_u1, 2 | (3 << 2) | (1 << 6));   // a2,a3; a0,a1
                        lo = _mm256_and_si256(t_u1, mask1);  // a2, a0
                        lo = _mm256_shuffle_epi8(table2, lo);   // a2lo, a0hi
                        hi = _mm256_srli_epi64(_mm256_and_si256(t_u1, mask2), 4);    // a3, a1
                        hi = _mm256_shuffle_epi8(table3, hi);   // a3lo, a1hi
                        t_u1 = _mm256_xor_si256(tmp, _mm256_xor_si256(hi, lo));

                        // + t_u0
                        t_u1 = _mm256_xor_si256(t_u0, t_u1);
                        _mm256_storeu_si256((__m256i *)&(inputs_sgm_share_v[e][u][k * 32]), t_u1);
                    }

                    // field::pair_mul(AV[e][u1], AG[e][u1], r.data, r.data, &AV[e][u1], &AG[e][u1]);
                    // AV[e][u] = AV[e][u0] ^ AV[e][u1];
                    // AG[e][u] = AG[e][u0] ^ AG[e][u1];
                    // AX[e][u] = AX[e][u0] ^ field::mul(AX[e][u1], r.data);
                    field::scale_mul(AV[e][u1], AG[e][u1], AX[e][u1], r.data,
                            AV[e][u0], AG[e][u0], AX[e][u0],
                            &AV[e][u], &AG[e][u], &AX[e][u]);
                    // AV[e][u] = AV[e][u0] ^ field::mul(AV[e][u1], r.data);
                    // AG[e][u] = AG[e][u0] ^ field::mul(AG[e][u1], r.data);
                    // AX[e][u] = AX[e][u0] ^ field::mul(AX[e][u1], r.data);
                }
            // else
            //     for (int uu = 0; uu < mask / 16; uu++) {
            //         for (int u = 0; u < 16; u++) {
            //             int u0 = (uu * 16 + u) << 1;
            //             int u1 = u0 + 1;
            //             // for (size_t p = 0; p < instance.num_MPC_parties; p++)
            //             //     inputs_sgm_share[e][u][p] = inputs_sgm_share[e][u0][p] + (inputs_sgm_share[e][u1][p] - inputs_sgm_share[e][u0][p]) * r;
            //             for (size_t k = 0; k < instance.num_MPC_parties / 16; k++) {
            //                 __m256i lo, hi, tmp;
            //                 __m256i t_u0 = _mm256_loadu_si256((__m256i *)&(inputs_sgm_share_v[e][u0][k * 32]));
            //                 __m256i t_u1 = _mm256_loadu_si256((__m256i *)&(inputs_sgm_share_v[e][u1][k * 32]));
            //                 t_u1 = _mm256_xor_si256(t_u0, t_u1);

            //                 // (t_u1 + t_u0) * r
            //                 lo = _mm256_and_si256(t_u1, mask1);  // a0, a2
            //                 lo = _mm256_shuffle_epi8(table0, lo);   // a0lo, a2hi
            //                 hi = _mm256_srli_epi64(_mm256_and_si256(t_u1, mask2), 4);    // a1, a3
            //                 hi = _mm256_shuffle_epi8(table1, hi);   // a1lo, a3hi
            //                 tmp = _mm256_xor_si256(hi, lo);
            //                 t_u1 = _mm256_permute4x64_epi64(t_u1, 2 | (3 << 2) | (1 << 6));   // a2,a3; a0,a1
            //                 lo = _mm256_and_si256(t_u1, mask1);  // a2, a0
            //                 lo = _mm256_shuffle_epi8(table2, lo);   // a2lo, a0hi
            //                 hi = _mm256_srli_epi64(_mm256_and_si256(t_u1, mask2), 4);    // a3, a1
            //                 hi = _mm256_shuffle_epi8(table3, hi);   // a3lo, a1hi
            //                 t_u1 = _mm256_xor_si256(tmp, _mm256_xor_si256(hi, lo));

            //                 // + t_u0
            //                 t_u1 = _mm256_xor_si256(t_u0, t_u1);
            //                 _mm256_storeu_si256((__m256i *)&(inputs_sgm_share_v[e][uu * 16 + u][k * 32]), t_u1);
            //             }
            //         }

            //         __m256i lo, hi, tmp;
            //         __m256i s_u0, s_u1, v_u0, v_u1;
            //         // AV
            //         s_u0 = _mm256_loadu_si256((__m256i *)&(AV[e][uu * 16 * 2]));
            //         s_u1 = _mm256_loadu_si256((__m256i *)&(AV[e][uu * 16 * 2 + 16]));
            //         v_u0 = _mm256_packus_epi32(
            //                 _mm256_and_si256(s_u0, _mm256_set1_epi32(0xffff)),
            //                 _mm256_and_si256(s_u1, _mm256_set1_epi32(0xffff)));
            //         v_u0 = _mm256_permute4x64_epi64(v_u0, 0b11011000);
            //         v_u1 = _mm256_packus_epi32(
            //                 _mm256_srli_epi32(s_u0, 16),
            //                 _mm256_srli_epi32(s_u1, 16));
            //         v_u1 = _mm256_permute4x64_epi64(v_u1, 0b11011000);
            //         // to alt mapping
            //         lo = _mm256_packus_epi16(_mm256_and_si256(v_u1, _mm256_set1_epi16(0xff)), _mm256_set1_epi64x(0));
            //         lo = _mm256_permute4x64_epi64(lo, 0b11011000);
            //         hi = _mm256_packus_epi16(_mm256_set1_epi64x(0), _mm256_srli_epi16(v_u1, 8));
            //         hi = _mm256_permute4x64_epi64(hi, 0b11011000);
            //         v_u1 = _mm256_xor_si256(lo, hi);

            //         // u1 * r
            //         lo = _mm256_and_si256(v_u1, mask1);  // a0, a2
            //         lo = _mm256_shuffle_epi8(table0, lo);   // a0lo, a2hi
            //         hi = _mm256_srli_epi64(_mm256_and_si256(v_u1, mask2), 4);    // a1, a3
            //         hi = _mm256_shuffle_epi8(table1, hi);   // a1lo, a3hi
            //         tmp = _mm256_xor_si256(hi, lo);
            //         v_u1 = _mm256_permute4x64_epi64(v_u1, 2 | (3 << 2) | (1 << 6));   // a2,a3; a0,a1
            //         lo = _mm256_and_si256(v_u1, mask1);  // a2, a0
            //         lo = _mm256_shuffle_epi8(table2, lo);   // a2lo, a0hi
            //         hi = _mm256_srli_epi64(_mm256_and_si256(v_u1, mask2), 4);    // a3, a1
            //         hi = _mm256_shuffle_epi8(table3, hi);   // a3lo, a1hi
            //         v_u1 = _mm256_xor_si256(tmp, _mm256_xor_si256(hi, lo));

            //         // to standard mapping
            //         v_u1 = _mm256_unpacklo_epi8(
            //                 _mm256_permute4x64_epi64(v_u1, 0b11011000),
            //                 _mm256_permute4x64_epi64(v_u1, 0b01110010));

            //         v_u1 = _mm256_xor_si256(v_u0, v_u1);

            //         _mm256_storeu_si256((__m256i *)&(AV[e][uu * 16]), v_u1);

            //         // AG
            //         s_u0 = _mm256_loadu_si256((__m256i *)&(AG[e][uu * 16 * 2]));
            //         s_u1 = _mm256_loadu_si256((__m256i *)&(AG[e][uu * 16 * 2 + 16]));
            //         v_u0 = _mm256_packus_epi32(
            //                 _mm256_and_si256(s_u0, _mm256_set1_epi32(0xffff)),
            //                 _mm256_and_si256(s_u1, _mm256_set1_epi32(0xffff)));
            //         v_u0 = _mm256_permute4x64_epi64(v_u0, 0b11011000);
            //         v_u1 = _mm256_packus_epi32(
            //                 _mm256_srli_epi32(s_u0, 16),
            //                 _mm256_srli_epi32(s_u1, 16));
            //         v_u1 = _mm256_permute4x64_epi64(v_u1, 0b11011000);
            //         // to alt mapping
            //         lo = _mm256_packus_epi16(_mm256_and_si256(v_u1, _mm256_set1_epi16(0xff)), _mm256_set1_epi64x(0));
            //         lo = _mm256_permute4x64_epi64(lo, 0b11011000);
            //         hi = _mm256_packus_epi16(_mm256_set1_epi64x(0), _mm256_srli_epi16(v_u1, 8));
            //         hi = _mm256_permute4x64_epi64(hi, 0b11011000);
            //         v_u1 = _mm256_xor_si256(lo, hi);

            //         // u1 * r
            //         lo = _mm256_and_si256(v_u1, mask1);  // a0, a2
            //         lo = _mm256_shuffle_epi8(table0, lo);   // a0lo, a2hi
            //         hi = _mm256_srli_epi64(_mm256_and_si256(v_u1, mask2), 4);    // a1, a3
            //         hi = _mm256_shuffle_epi8(table1, hi);   // a1lo, a3hi
            //         tmp = _mm256_xor_si256(hi, lo);
            //         v_u1 = _mm256_permute4x64_epi64(v_u1, 2 | (3 << 2) | (1 << 6));   // a2,a3; a0,a1
            //         lo = _mm256_and_si256(v_u1, mask1);  // a2, a0
            //         lo = _mm256_shuffle_epi8(table2, lo);   // a2lo, a0hi
            //         hi = _mm256_srli_epi64(_mm256_and_si256(v_u1, mask2), 4);    // a3, a1
            //         hi = _mm256_shuffle_epi8(table3, hi);   // a3lo, a1hi
            //         v_u1 = _mm256_xor_si256(tmp, _mm256_xor_si256(hi, lo));

            //         // to standard mapping
            //         v_u1 = _mm256_unpacklo_epi8(
            //                 _mm256_permute4x64_epi64(v_u1, 0b11011000),
            //                 _mm256_permute4x64_epi64(v_u1, 0b01110010));

            //         v_u1 = _mm256_xor_si256(v_u0, v_u1);

            //         _mm256_storeu_si256((__m256i *)&(AG[e][uu * 16]), v_u1);

            //         // AX
            //         s_u0 = _mm256_loadu_si256((__m256i *)&(AX[e][uu * 16 * 2]));
            //         s_u1 = _mm256_loadu_si256((__m256i *)&(AX[e][uu * 16 * 2 + 16]));
            //         v_u0 = _mm256_packus_epi32(
            //                 _mm256_and_si256(s_u0, _mm256_set1_epi32(0xffff)),
            //                 _mm256_and_si256(s_u1, _mm256_set1_epi32(0xffff)));
            //         v_u0 = _mm256_permute4x64_epi64(v_u0, 0b11011000);
            //         v_u1 = _mm256_packus_epi32(
            //                 _mm256_srli_epi32(s_u0, 16),
            //                 _mm256_srli_epi32(s_u1, 16));
            //         v_u1 = _mm256_permute4x64_epi64(v_u1, 0b11011000);
            //         // to alt mapping
            //         lo = _mm256_packus_epi16(_mm256_and_si256(v_u1, _mm256_set1_epi16(0xff)), _mm256_set1_epi64x(0));
            //         lo = _mm256_permute4x64_epi64(lo, 0b11011000);
            //         hi = _mm256_packus_epi16(_mm256_set1_epi64x(0), _mm256_srli_epi16(v_u1, 8));
            //         hi = _mm256_permute4x64_epi64(hi, 0b11011000);
            //         v_u1 = _mm256_xor_si256(lo, hi);

            //         // u1 * r
            //         lo = _mm256_and_si256(v_u1, mask1);  // a0, a2
            //         lo = _mm256_shuffle_epi8(table0, lo);   // a0lo, a2hi
            //         hi = _mm256_srli_epi64(_mm256_and_si256(v_u1, mask2), 4);    // a1, a3
            //         hi = _mm256_shuffle_epi8(table1, hi);   // a1lo, a3hi
            //         tmp = _mm256_xor_si256(hi, lo);
            //         v_u1 = _mm256_permute4x64_epi64(v_u1, 2 | (3 << 2) | (1 << 6));   // a2,a3; a0,a1
            //         lo = _mm256_and_si256(v_u1, mask1);  // a2, a0
            //         lo = _mm256_shuffle_epi8(table2, lo);   // a2lo, a0hi
            //         hi = _mm256_srli_epi64(_mm256_and_si256(v_u1, mask2), 4);    // a3, a1
            //         hi = _mm256_shuffle_epi8(table3, hi);   // a3lo, a1hi
            //         v_u1 = _mm256_xor_si256(tmp, _mm256_xor_si256(hi, lo));

            //         // to standard mapping
            //         v_u1 = _mm256_unpacklo_epi8(
            //                 _mm256_permute4x64_epi64(v_u1, 0b11011000),
            //                 _mm256_permute4x64_epi64(v_u1, 0b01110010));

            //         v_u1 = _mm256_xor_si256(v_u0, v_u1);

            //         _mm256_storeu_si256((__m256i *)&(AX[e][uu * 16]), v_u1);

            //         // for (int u = 0; u < 16; u++) {
            //         //     int u0 = (uu * 16 + u) << 1;
            //         //     int u1 = u0 + 1;
            //         //     AV[e][uu * 16 + u] = AV[e][u0] ^ field::mul(AV[e][u1], r.data);
            //         //     AG[e][uu * 16 + u] = AG[e][u0] ^ field::mul(AG[e][u1], r.data);
            //         //     AX[e][uu * 16 + u] = AX[e][u0] ^ field::mul(AX[e][u1], r.data);
            //         // }
            //     }

            // update pred
            field::GF2E one(1);
            if (i < (bit_len - 1))
                pred[e] *= tau[i][e] * rho[i][e] + (tau[i][e] + rho[i][e] + one) * (r + one);
            else
                pred[e] *= tau[i][e] * rho[i][e] + (tau[i][e] + rho[i][e] + one) * r;
        }
    }

#ifdef TIMING
    tmp_time = timing_read(&ctx);
    printf("gkr phase 2 loop time: %ld\n", tmp_time - start_time);
    start_time = timing_read(&ctx);
#endif

    for (size_t e = 0; e < instance.num_rounds; e++) {
        Vs[e][0] = AV[e][0];
        for (size_t k = 0; k < instance.num_MPC_parties / 16; k++)
            for (size_t p = 0; p < 16; p++) {
                auto Vr = Vr_share.get(e, k * 16 + p);
                auto Vs = Vs_share.get(e, k * 16 + p);
                auto fr = fr_share.get(e, k * 16 + p);
                fr[0] = field::GF2E(fr_shares_v[e][k * 32 + p] | ((uint16_t)fr_shares_v[e][k * 32 + p + 16] << 8));
                Vr[0] = field::GF2E(inputs_rho_share_v[e][0][k * 32 + p] | ((uint16_t)inputs_rho_share_v[e][0][k * 32 + p + 16] << 8));
                Vs[0] = field::GF2E(inputs_sgm_share_v[e][0][k * 32 + p] | ((uint16_t)inputs_sgm_share_v[e][0][k * 32 + p + 16] << 8));
                // Vr[0] = inputs_rho_share[e][0][k * 16 + p];
                // Vs[0] = inputs_sgm_share[e][0][k * 16 + p];
            }
    }

#ifdef TIMING
    tmp_time = timing_read(&ctx);
    printf("gkr final time: %ld\n", tmp_time - start_time);
    start_time = timing_read(&ctx);
#endif

    // next phase: prove that
    // sum(broadcast[0]) = pred * sum(broadcast[1])(1 - sum(broadcast[1]) * sum(broadcast[2]))
    // a + pred * b * (1 + b * c) = 0

    // do a sanity check here
    // for (size_t e = 0; e < instance.num_rounds; e++) {
    //     field::GF2E tmp0(0);
    //     field::GF2E tmp1(0);
    //     field::GF2E tmp2(0);
    //     for (size_t p = 0; p < instance.num_MPC_parties; p++) {
    //         auto a = fr_share.get(e, p);
    //         auto b = Vr_share.get(e, p);
    //         auto c = Vs_share.get(e, p);
    //         tmp0 += a[0];
    //         tmp1 += b[0];
    //         tmp2 += c[0];
    //     }
    //     if (tmp1 != Vr[e][0])
    //         throw std::runtime_error("sanity check Vr failed");
    //     if (tmp2 != Vs[e][0])
    //         throw std::runtime_error("sanity check Vs failed");
    //     if (tmp0 != fr[e][0])
    //         throw std::runtime_error("sanity check fr failed");
    //     if (tmp0 != pred[e] * tmp1 * (field::GF2E(1) - tmp1 * tmp2))
    //         throw std::runtime_error("sanity check GKR failed");
    // }

    // expand challenge hash to M * m1 values
    // std::vector<std::vector<field::GF2E>> r_ejs = phase_1_expand(instance, h_1);

    /////////////////////////////////////////////////////////////////////////////
    // phase 3: commit to the checking polynomials
    /////////////////////////////////////////////////////////////////////////////

    // a vector of the first m2+1 field elements for interpolation
    // m2 = 1
    std::vector<field::GF2E> x_values_for_interpolation_zero_to_m2 = field::get_first_n_field_elements(1 + 1);
    std::vector<std::vector<field::GF2E>> precomputation_for_zero_to_m2 = precompute_lagrange_polynomials(x_values_for_interpolation_zero_to_m2);

    std::vector<field::GF2E> x_values_for_interpolation_zero_to_3m2 = field::get_first_n_field_elements(3 * 1 + 1);
    std::vector<std::vector<field::GF2E>> precomputation_for_zero_to_3m2 = precompute_lagrange_polynomials(x_values_for_interpolation_zero_to_3m2);

    std::vector<std::vector<std::vector<field::GF2E>>> fr_poly_share(instance.num_rounds);
    std::vector<std::vector<std::vector<field::GF2E>>> Vr_poly_share(instance.num_rounds);
    std::vector<std::vector<std::vector<field::GF2E>>> Vs_poly_share(instance.num_rounds);
    std::vector<std::vector<std::vector<field::GF2E>>> P_share(instance.num_rounds);
    std::vector<std::vector<field::GF2E>> fr_poly(instance.num_rounds);
    std::vector<std::vector<field::GF2E>> Vr_poly(instance.num_rounds);
    std::vector<std::vector<field::GF2E>> Vs_poly(instance.num_rounds);
    std::vector<std::vector<field::GF2E>> P_poly(instance.num_rounds);

    std::vector<std::vector<field::GF2E>> P(instance.num_rounds);
    std::vector<std::vector<field::GF2E>> P_deltas(instance.num_rounds);
    std::vector<field::GF2E> one(1, field::GF2E(1));

    for (size_t e = 0; e < instance.num_rounds; e++) {
        fr_poly_share[e].resize(instance.num_MPC_parties);
        Vr_poly_share[e].resize(instance.num_MPC_parties);
        Vs_poly_share[e].resize(instance.num_MPC_parties);
        P_share[e].resize(instance.num_MPC_parties);
        P[e].resize(2 * 1 + 1);
        P_deltas[e].resize(2 * 1 + 1);

        for (size_t p = 0; p < instance.num_MPC_parties; p++) {
            auto a = fr_share.get(e, p);
            auto b = Vr_share.get(e, p);
            auto c = Vs_share.get(e, p);

            auto random_share = random_tapes.get_bytes(e, p,
                        instance.aes_params.key_size +
                        instance.aes_params.block_size * instance.aes_params.num_blocks +
                        instance.aes_params.num_sboxes +
                        3 * bit_len * instance.lambda +
                        2 * bit_len * instance.lambda,
                        3 * instance.lambda + 3 * instance.lambda);

            a[1].from_bytes(random_share.data());
            b[1].from_bytes(random_share.data() + instance.lambda);
            c[1].from_bytes(random_share.data() + 2 * instance.lambda);
            fr[e][1] += a[1];
            Vr[e][1] += b[1];
            Vs[e][1] += c[1];
            fr_poly_share[e][p] = interpolate_with_precomputation(precomputation_for_zero_to_m2, a);
            Vr_poly_share[e][p] = interpolate_with_precomputation(precomputation_for_zero_to_m2, b);
            Vs_poly_share[e][p] = interpolate_with_precomputation(precomputation_for_zero_to_m2, c);

            P_share[e][p].resize(3 * 1 + 1);
            P_share[e][p][1].from_bytes(random_share.data() + 3 * instance.lambda);
            P_share[e][p][2].from_bytes(random_share.data() + 4 * instance.lambda);
            P_share[e][p][3].from_bytes(random_share.data() + 5 * instance.lambda);
            P_deltas[e][0] += P_share[e][p][1];
            P_deltas[e][1] += P_share[e][p][2];
            P_deltas[e][2] += P_share[e][p][3];
        }
        fr_poly[e] = interpolate_with_precomputation(precomputation_for_zero_to_m2, fr[e]);
        Vr_poly[e] = interpolate_with_precomputation(precomputation_for_zero_to_m2, Vr[e]);
        Vs_poly[e] = interpolate_with_precomputation(precomputation_for_zero_to_m2, Vs[e]);
        auto P = fr_poly[e] + pred[e] * Vr_poly[e] * (one + Vr_poly[e] * Vs_poly[e]);

        for (size_t k = 1; k <= 3 * 1; k++) {
            // calculate offset
            field::GF2E k_element = x_values_for_interpolation_zero_to_3m2[k];
            P_deltas[e][k - 1] = eval(P, k_element) - P_deltas[e][k - 1];
            // adjust first share
            P_share[e][0][k] += P_deltas[e][k - 1];
        }
    }

#ifdef TIMING
    tmp_time = timing_read(&ctx);
    printf("linear pcp time: %ld\n", tmp_time - start_time);
    start_time = timing_read(&ctx);
#endif

    /////////////////////////////////////////////////////////////////////////////
    // phase 4: challenge the checking polynomials
    /////////////////////////////////////////////////////////////////////////////

    std::vector<uint8_t> h_2 = phase_2_commitment(instance, salt, h_i, P_deltas);

    // expand challenge hash to M values

    std::vector<field::GF2E> forbidden_challenge_values = field::get_first_n_field_elements(1);
    std::vector<field::GF2E> R_es = phase_2_expand(instance, h_2, forbidden_challenge_values);

    /////////////////////////////////////////////////////////////////////////////
    // phase 5: commit to the views of the checking protocol
    /////////////////////////////////////////////////////////////////////////////

    // std::vector<field::GF2E> d(instance.num_rounds);
    std::vector<std::vector<field::GF2E>> d_shares(instance.num_rounds);

    std::vector<field::GF2E> a(instance.num_rounds);
    std::vector<field::GF2E> b(instance.num_rounds);
    std::vector<field::GF2E> c(instance.num_rounds);
    std::vector<std::vector<field::GF2E>> a_shares(instance.num_rounds);
    std::vector<std::vector<field::GF2E>> b_shares(instance.num_rounds);
    std::vector<std::vector<field::GF2E>> c_shares(instance.num_rounds);

    std::vector<field::GF2E> lagrange_polys_evaluated_at_Re_m2(1 + 1);
    std::vector<field::GF2E> lagrange_polys_evaluated_at_Re_3m2(3 * 1 + 1);

    for (size_t e = 0; e < instance.num_rounds; e++) {
        a_shares[e].resize(instance.num_MPC_parties);
        b_shares[e].resize(instance.num_MPC_parties);
        c_shares[e].resize(instance.num_MPC_parties);
        d_shares[e].resize(instance.num_MPC_parties);
        for (size_t k = 0; k < 1 + 1; k++)
            lagrange_polys_evaluated_at_Re_m2[k] = eval(precomputation_for_zero_to_m2[k], R_es[e]);
        for (size_t k = 0; k < 3 * 1 + 1; k++)
            lagrange_polys_evaluated_at_Re_3m2[k] = eval(precomputation_for_zero_to_3m2[k], R_es[e]);

        for (size_t p = 0; p < instance.num_MPC_parties; p++) {
            auto fr   = fr_share.get(e, p);
            auto Vr = Vr_share.get(e, p);
            auto Vs = Vs_share.get(e, p);
            a_shares[e][p] = dot_product(lagrange_polys_evaluated_at_Re_m2, fr);
            b_shares[e][p] = dot_product(lagrange_polys_evaluated_at_Re_m2, Vr);
            c_shares[e][p] = dot_product(lagrange_polys_evaluated_at_Re_m2, Vs);

          // compute c_e^i
            d_shares[e][p] = dot_product(lagrange_polys_evaluated_at_Re_3m2, P_share[e][p]);

        }

        // open d_e and a,b,c values
        for (size_t p = 0; p < instance.num_MPC_parties; p++) {
            a[e] += a_shares[e][p];
            b[e] += b_shares[e][p];
            c[e] += c_shares[e][p];
        }
    }

    std::vector<uint8_t> h_3 = phase_3_commitment(
            instance, salt, h_2, d_shares, a, a_shares, b, b_shares, c, c_shares);

    int random_offset = instance.aes_params.key_size +
                        instance.aes_params.block_size * instance.aes_params.num_blocks +
                        instance.aes_params.num_sboxes +
                        3 * bit_len * instance.lambda +
                        2 * bit_len * instance.lambda +
                        3 * instance.lambda + 3 * instance.lambda;

    /////////////////////////////////////////////////////////////////////////////
    // phase GROUP 1: gkr for multiplication of all parties' ct/pt diff
    /////////////////////////////////////////////////////////////////////////////

#ifdef TIMING
    tmp_time = timing_read(&ctx);
    printf("phase4&5 time: %ld\n", tmp_time - start_time);
    start_time = timing_read(&ctx);
#endif

    h_i = h_3;
    std::vector<uint8_t> random_combinations = phase_group1_expand(instance, h_3, instance.num_rounds * gkey.size());

    // inputs of the second gkr of each party
    // RepByteContainer rep_shared_keytest(
    //         instance.num_rounds, instance.num_MPC_parties,
    //         gkey.size());
    // std::vector<std::vector<std::vector<field::GF2E>>> keytest_share(instance.num_rounds);
    std::vector<std::vector<std::vector<uint8_t>>> keytest_share_v(instance.num_rounds);
    std::vector<std::vector<std::vector<field::GF2E>>> gate_outputs(instance.num_rounds);

    for (size_t e = 0; e < instance.num_rounds; e++) {
        gate_outputs[e].resize(depth + 1);
        // keytest_share[e].resize(gkey.size());
        keytest_share_v[e].resize(gkey.size());
        gate_outputs[e][depth].resize(gkey.size());
        auto& keytest = gate_outputs[e][depth];

        for (size_t g = 0; g < gkey.size(); g++) {
            // keytest_share[e][g].resize(instance.num_MPC_parties);
            keytest_share_v[e][g].resize(instance.num_MPC_parties);
            int offset = (e * gkey.size() + g) * 2 * instance.aes_params.block_size * instance.aes_params.num_blocks;
            keytest[g] = 0;
            int j = offset;
            unsigned int i = 0;
            for (; i < instance.aes_params.block_size * instance.aes_params.num_blocks; i++)
                keytest[g] += field::lift_uint8_t(random_combinations[j++]) * (field::lift_uint8_t(pt_ct[i]) - field::lift_uint8_t(gkey[g].second[i]));
            for (; i < 2 * instance.aes_params.block_size * instance.aes_params.num_blocks; i++)
                keytest[g] += field::lift_uint8_t(random_combinations[j++]) * (field::lift_uint8_t(pt_ct[i]) - field::lift_uint8_t(gkey[g].second[i]));
            // printf("gate_outputs[%ld][%d][%d]=0x%08x\n", e, depth, g, (uint32_t)(gate_outputs[e][depth][g].data));
            for (size_t k = 0; k < instance.num_MPC_parties / 16; k++) {
                __m256i res = _mm256_setzero_si256();
                for (unsigned int i = 0; i < instance.aes_params.block_size * instance.aes_params.num_blocks; i++) {
                    uint8_t t[32];
                    auto r0 = random_combinations[offset + i];
                    auto r1 = random_combinations[offset + i + instance.aes_params.block_size * instance.aes_params.num_blocks];
                    for (int p = 0; p < 16; p++) {
                        auto pt_shares = rep_shared_pt.get(e, k * 16 + p);
                        auto ct_shares = rep_shared_ct.get(e, k * 16 + p);
                        uint16_t pt, ct;
                        if (k == 0 && p == 0) {
                            pt = pt_shares[i] ^ gkey[g].second[i];
                            ct = ct_shares[i] ^ gkey[g].second[i + instance.aes_params.block_size * instance.aes_params.num_blocks];
                        } else {
                            pt = pt_shares[i];
                            ct = ct_shares[i];
                        }
                        t[p] = pt;
                        t[p + 16] = ct;
                    }
                    const __m256i mask1 = _mm256_set1_epi8(0x0f);
                    const __m256i mask2 = _mm256_set1_epi8(0xf0);
                    const __m256i table0 = _mm256_loadu_si256((__m256i *)&(gf256_mul_lut[r0 ^ (r1 << 8)][0]));
                    const __m256i table1 = _mm256_loadu_si256((__m256i *)&(gf256_mul_lut[r0 ^ (r1 << 8)][8]));
                    __m256i t_t = _mm256_loadu_si256((__m256i *)&t);

                    __m256i lo, hi;
                    lo = _mm256_and_si256(t_t, mask1);
                    lo = _mm256_shuffle_epi8(table0, lo);
                    hi = _mm256_and_si256(t_t, mask2);
                    hi = _mm256_srli_epi64(hi, 4);
                    hi = _mm256_shuffle_epi8(table1, hi);
                    t_t = _mm256_xor_si256(hi, lo);

                    res = _mm256_xor_si256(res, t_t);
                }
                _mm_storeu_si128((__m128i *)&(keytest_share_v[e][g][k * 16]),
                        _mm_xor_si128(
                            _mm256_castsi256_si128(res),
                            _mm256_extracti128_si256(res, 1)));
            }

            // {
            //     size_t p = 0;
            //     auto pt_shares = rep_shared_pt.get(e, p);
            //     auto ct_shares = rep_shared_ct.get(e, p);
            //     int j = offset;
            //     for (unsigned int i = 0; i < instance.aes_params.block_size * instance.aes_params.num_blocks; i++)
            //         keytest_share[e][g][p] +=
            //             field::lift_uint8_t(random_combinations[j++]) *
            //             field::lift_uint8_t(pt_shares[i] ^ gkey[g].second[i]);
            //     for (unsigned int i = 0; i < instance.aes_params.block_size * instance.aes_params.num_blocks; i++)
            //         keytest_share[e][g][p] +=
            //             field::lift_uint8_t(random_combinations[j++]) *
            //             field::lift_uint8_t(ct_shares[i] ^ gkey[g].second[i + instance.aes_params.block_size * instance.aes_params.num_blocks]);
            // }
            // for (size_t p = 1; p < instance.num_MPC_parties; p++) {
            //     auto pt_shares = rep_shared_pt.get(e, p);
            //     auto ct_shares = rep_shared_ct.get(e, p);
            //     int j = offset;
            //     for (unsigned int i = 0; i < instance.aes_params.block_size * instance.aes_params.num_blocks; i++)
            //         keytest_share[e][g][p] +=
            //             field::lift_uint8_t(random_combinations[j++]) *
            //             field::lift_uint8_t(pt_shares[i]);
            //     for (unsigned int i = 0; i < instance.aes_params.block_size * instance.aes_params.num_blocks; i++)
            //         keytest_share[e][g][p] +=
            //             field::lift_uint8_t(random_combinations[j++]) *
            //             field::lift_uint8_t(ct_shares[i]);
            // }

            // for (size_t k = 0; k < instance.num_MPC_parties / 16; k++) {
            //     for (int p = 0; p < 16; p++) {
            //         if (keytest_share[e][g][k * 16 + p] !=
            //                 field::lift_uint8_t(
            //                     keytest_share_v[e][g][k * 16 + p]))
            //             printf("wrong %ld\n", k*16+p);
            //     }
            // }
        }
        for (int d = depth - 1; d >= 0; d--) {
            gate_outputs[e][d].resize(1 << d);
            for (int i = 0; i < (1 << d); i++) {
                gate_outputs[e][d][i] = gate_outputs[e][d+1][2*i] * gate_outputs[e][d+1][2*i+1];
                // printf("gate_outputs[%ld][%d][%d]=0x%08x\n", e, d, i, (uint32_t)(gate_outputs[e][d][i].data));
            }
        }
    }

#ifdef TIMING
    tmp_time = timing_read(&ctx);
    printf("GGKR input comb time: %ld\n", tmp_time - start_time);
    start_time = timing_read(&ctx);
#endif

    std::vector<std::vector<field::GF2E>> g_Vr(instance.num_rounds);
    std::vector<std::vector<field::GF2E>> g_Vs(instance.num_rounds);
    std::vector<std::vector<field::GF2E>> g_fr(instance.num_rounds);
    std::vector<std::vector<field::GF2E>> g_pred(instance.num_rounds);
    std::vector<std::vector<field::GF2E>> t_pred(instance.num_rounds);
    std::vector<std::vector<field::GF2E>> g_Vr_deltas(depth - 1);
    std::vector<std::vector<field::GF2E>> g_Vs_deltas(depth - 1);
    // std::vector<std::vector<std::vector<field::GF2E>>> g_Vr_shares(instance.num_rounds);
    // std::vector<std::vector<std::vector<field::GF2E>>> g_Vs_shares(instance.num_rounds);
    // std::vector<std::vector<std::vector<field::GF2E>>> g_fr_shares(instance.num_rounds);
    std::vector<std::vector<std::vector<uint8_t>>> g_Vr_shares_v(instance.num_rounds);
    std::vector<std::vector<std::vector<uint8_t>>> g_Vs_shares_v(instance.num_rounds);
    std::vector<std::vector<std::vector<uint8_t>>> g_fr_shares_v(instance.num_rounds);

    std::vector<std::vector<field::GF2E>> g_Rho(instance.num_rounds);
    std::vector<std::vector<field::GF2E>> g_Sgm(instance.num_rounds);
    std::vector<std::vector<field::GF2E>> g_Rho_old(instance.num_rounds);
    std::vector<std::vector<field::GF2E>> g_Sgm_old(instance.num_rounds);
    // std::vector<std::vector<uint16_t>> g_Rho(instance.num_rounds);
    // std::vector<std::vector<uint16_t>> g_Sgm(instance.num_rounds);
    // std::vector<std::vector<uint16_t>> g_Rho_old(instance.num_rounds);
    // std::vector<std::vector<uint16_t>> g_Sgm_old(instance.num_rounds);

    std::vector<std::vector<field::GF2E>> g_AG(instance.num_rounds);
    std::vector<std::vector<field::GF2E>> g_AV(instance.num_rounds);
    // std::vector<std::vector<uint16_t>> g_AG(instance.num_rounds);
    // std::vector<std::vector<uint16_t>> g_AV(instance.num_rounds);

    std::vector<std::vector<std::vector<std::array<std::vector<uint8_t>, 4>>>> g_coef_shares_v(instance.num_rounds);
    std::vector<std::vector<std::vector<std::array<std::vector<field::GF2E>, 4>>>> g_coef_shares(instance.num_rounds);
    std::vector<std::vector<std::vector<std::array<field::GF2E, 2>>>> g_coef1(depth);
    std::vector<std::vector<std::vector<std::vector<field::GF2E>>>>   g_coef1_deltas(depth);
    std::vector<std::vector<std::vector<std::array<field::GF2E, 2>>>> g_coef2(depth);
    std::vector<std::vector<std::vector<std::vector<field::GF2E>>>>   g_coef2_deltas(depth);

    std::vector<std::vector<std::vector<field::GF2E>>> g_rho(depth);
    std::vector<std::vector<std::vector<field::GF2E>>> g_sgm(depth);
    std::vector<std::vector<std::vector<field::GF2E>>> g_scalar(depth - 1);

    std::vector<std::vector<std::vector<field::GF2E>>> g_fr_poly_share(instance.num_rounds);
    std::vector<std::vector<std::vector<field::GF2E>>> g_Vr_poly_share(instance.num_rounds);
    std::vector<std::vector<std::vector<field::GF2E>>> g_Vs_poly_share(instance.num_rounds);
    std::vector<std::vector<std::vector<field::GF2E>>> g_P_share(instance.num_rounds);
    std::vector<std::vector<field::GF2E>> g_fr_poly(instance.num_rounds);
    std::vector<std::vector<field::GF2E>> g_Vr_poly(instance.num_rounds);
    std::vector<std::vector<field::GF2E>> g_Vs_poly(instance.num_rounds);
    std::vector<std::vector<field::GF2E>> g_P_poly(instance.num_rounds);
    std::vector<std::vector<field::GF2E>> g_P(instance.num_rounds);
    std::vector<std::vector<field::GF2E>> g_P_deltas(instance.num_rounds);

    std::vector<field::GF2E> x_values_for_interpolation_zero_to_d = field::get_first_n_field_elements(depth + 1);
    std::vector<std::vector<field::GF2E>> precomputation_for_zero_to_d = precompute_lagrange_polynomials(x_values_for_interpolation_zero_to_d);
    std::vector<field::GF2E> x_values_for_interpolation_zero_to_2d = field::get_first_n_field_elements(2 * depth + 1);
    std::vector<std::vector<field::GF2E>> precomputation_for_zero_to_2d = precompute_lagrange_polynomials(x_values_for_interpolation_zero_to_2d);

    {
        //init vectors
        // TODO init Vr Vs
        {
            for (size_t e = 0; e < instance.num_rounds; e++) {
                g_coef_shares[e].resize(depth);
                g_coef_shares_v[e].resize(depth);
                g_pred[e].resize(depth);
                t_pred[e].resize(depth);
                g_fr[e].resize(depth + 1);
                g_Vr[e].resize(depth + 1);
                g_Vs[e].resize(depth + 1);
                // g_fr_shares[e].resize(instance.num_MPC_parties);
                // g_Vr_shares[e].resize(instance.num_MPC_parties);
                // g_Vs_shares[e].resize(instance.num_MPC_parties);
                for (int d = 0; d < depth; d++) {
                    int bit_len = d + 1;
                    g_pred[e][d] = field::GF2E(1);
                    t_pred[e][d] = field::GF2E(1);
                    g_coef_shares[e][d].resize(bit_len);
                    g_coef_shares_v[e][d].resize(bit_len);
                    for (int i = 0; i < bit_len; i++)
                        for (int j = 0; j < 4; j++) {
                            g_coef_shares[e][d][i][j].resize(instance.num_MPC_parties);
                            g_coef_shares_v[e][d][i][j].resize(instance.num_MPC_parties * 2);
                        }
                }
                // for (size_t p = 0; p < instance.num_MPC_parties; p++) {
                //     g_fr_shares[e][p].resize(depth + 1);
                //     g_Vr_shares[e][p].resize(depth + 1);
                //     g_Vs_shares[e][p].resize(depth + 1);
                // }
                g_fr_shares_v[e].resize(depth + 1);
                g_Vr_shares_v[e].resize(depth + 1);
                g_Vs_shares_v[e].resize(depth + 1);
                for (int d = 0; d < depth + 1; d++) {
                    g_fr_shares_v[e][d].resize(instance.num_MPC_parties * 2);
                    g_Vr_shares_v[e][d].resize(instance.num_MPC_parties * 2);
                    g_Vs_shares_v[e][d].resize(instance.num_MPC_parties * 2);
                }
            }
        }

#ifdef TIMING
    tmp_time = timing_read(&ctx);
    printf("GGKR init time: %ld\n", tmp_time - start_time);
    start_time = timing_read(&ctx);
#endif
        // output layer
        // phase 1
        // init AV, AG
        bit_len = 1;
        int d = 0;
        for (size_t e = 0; e < instance.num_rounds; e++) {
            g_AV[e] = gate_outputs[e][d + 1];
            g_AG[e].resize(1 << bit_len);
            g_Rho[e].resize(1 << (d + 1));
            g_Sgm[e].resize(1 << (d + 1));
            g_Rho[e][0] = field::GF2E(1);
            g_Sgm[e][0] = field::GF2E(1);
            for (int u = 0; u < (1 << (bit_len - 1)); u++)
                g_AG[e][2 * u] = g_AV[e][2 * u + 1];
        }

        g_coef1[d].resize(bit_len);
        g_coef1_deltas[d].resize(bit_len);
        for (int i = 0; i < bit_len; i++) {
            g_coef1[d][i].resize(instance.num_rounds);
            g_coef1_deltas[d][i].resize(instance.num_rounds);
            // whole computes
            for (size_t e = 0; e < instance.num_rounds; e++) {
                for (int u = 0; u < (1 << (bit_len - i)); u += 2) {
                    int one_u = u + 1;
                    g_AV[e][one_u] += g_AV[e][u];
                    g_AG[e][one_u] += g_AG[e][u];
                    g_coef1[d][i][e][0] += g_AV[e][u] * g_AG[e][u];
                    g_coef1[d][i][e][1] += g_AV[e][one_u] * g_AG[e][one_u];
                }
                g_fr[e][d] -= g_coef1[d][i][e][1];
            }
            // parties compute shares
            for (size_t e = 0; e < instance.num_rounds; e++) {
                g_coef1_deltas[d][i][e].resize(2);
                __m256i g_coef1_deltas_v[2];
                g_coef1_deltas_v[0] = _mm256_setzero_si256();
                g_coef1_deltas_v[1] = _mm256_setzero_si256();
                for (size_t k = 0; k < instance.num_MPC_parties / 16; k++) {
                    for (size_t p = 0; p < 16; p++) {
                        auto random_coef_share =
                            random_tapes.get_bytes(e, k * 16 + p,
                                    random_offset,
                                    2 * instance.lambda);
                        g_coef_shares_v[e][d][i][0][k * 32 + p] = *(uint8_t *)(random_coef_share.data());
                        g_coef_shares_v[e][d][i][1][k * 32 + p] = *(uint8_t *)(random_coef_share.data() + 2);
                        g_coef_shares_v[e][d][i][0][k * 32 + 16 + p] = *(uint8_t *)(random_coef_share.data() + 1);
                        g_coef_shares_v[e][d][i][1][k * 32 + 16 + p] = *(uint8_t *)(random_coef_share.data() + 3);

                    }
                    __m256i t_coef0 = _mm256_loadu_si256((__m256i *)&(g_coef_shares_v[e][d][i][0][k * 32]));
                    __m256i t_coef1 = _mm256_loadu_si256((__m256i *)&(g_coef_shares_v[e][d][i][1][k * 32]));
                    __m256i t_fr    = _mm256_loadu_si256((__m256i *)&(g_fr_shares_v[e][d][k * 32]));
                    t_fr = _mm256_xor_si256(t_coef1, t_fr);
                    _mm256_storeu_si256((__m256i *)&(g_fr_shares_v[e][d][k * 32]), t_fr);
                    g_coef1_deltas_v[0] = _mm256_xor_si256(g_coef1_deltas_v[0], t_coef0);
                    g_coef1_deltas_v[1] = _mm256_xor_si256(g_coef1_deltas_v[1], t_coef1);
                }
                g_coef1_deltas[d][i][e][0] = g_coef1[d][i][e][0] - field::GF2E(_mm256_hxor_epu16(g_coef1_deltas_v[0]));
                g_coef1_deltas[d][i][e][1] = g_coef1[d][i][e][1] - field::GF2E(_mm256_hxor_epu16(g_coef1_deltas_v[1]));

                g_coef_shares_v[e][d][i][0][0] ^= (g_coef1_deltas[d][i][e][0].data & 255);
                g_coef_shares_v[e][d][i][1][0] ^= (g_coef1_deltas[d][i][e][1].data & 255);
                g_coef_shares_v[e][d][i][0][16] ^= (g_coef1_deltas[d][i][e][0].data >> 8);
                g_coef_shares_v[e][d][i][1][16] ^= (g_coef1_deltas[d][i][e][1].data >> 8);

                g_fr_shares_v[e][d][0] ^= g_coef1_deltas[d][i][e][1].data & 255;
                g_fr_shares_v[e][d][16] ^= g_coef1_deltas[d][i][e][1].data >> 8;

                // field::GF2E delta0(0);
                // field::GF2E delta1(0);
                // for (size_t p = 0; p < instance.num_MPC_parties; p++) {
                //     auto random_coef_share =
                //         random_tapes.get_bytes(e, p,
                //                 random_offset,
                //                 2 * instance.lambda);
                //     g_coef_shares[e][d][i][0][p].from_bytes(random_coef_share.data());
                //     g_coef_shares[e][d][i][1][p].from_bytes(random_coef_share.data() + instance.lambda);
                //     g_fr_shares[e][p][d] -= g_coef_shares[e][d][i][1][p];
                //     delta0 += g_coef_shares[e][d][i][0][p];
                //     delta1 += g_coef_shares[e][d][i][1][p];
                // }
                // delta0 = g_coef1[d][i][e][0] - delta0;
                // delta1 = g_coef1[d][i][e][1] - delta1;
                // g_coef_shares[e][d][i][0][0] += delta0;
                // g_coef_shares[e][d][i][1][0] += delta1;
                // g_fr_shares[e][0][d] += delta1;

                // if (delta0 != g_coef1_deltas[d][i][e][0])
                //     printf("wtf delta0\n");
                // if (delta1 != g_coef1_deltas[d][i][e][1])
                //     printf("wtf delta1\n");
                // for (size_t k = 0; k < instance.num_MPC_parties / 16; k++) {
                //     for (int p = 0; p < 16; p++) {
                //         if (g_fr_shares_v[e][d][k * 32 + p] != (g_fr_shares[e][k * 16 + p][d].data & 255))
                //             printf("wtf fr lo\n");
                //         else if (g_fr_shares_v[e][d][k * 32 + p + 16] != (g_fr_shares[e][k * 16 + p][d].data >> 8))
                //             printf("wtf fr hi\n");
                //         else if (g_coef_shares_v[e][d][i][0][k * 32 + p] != (g_coef_shares[e][d][i][0][k * 16 + p].data & 255))
                //             printf("wtf c0 lo\n");
                //         else if (g_coef_shares_v[e][d][i][0][k * 32 + p + 16] != (g_coef_shares[e][d][i][0][k * 16 + p].data >> 8))
                //             printf("wtf c0 hi\n");
                //         else if (g_coef_shares_v[e][d][i][1][k * 32 + p] != (g_coef_shares[e][d][i][1][k * 16 + p].data & 255))
                //             printf("wtf c1 lo\n");
                //         else if (g_coef_shares_v[e][d][i][1][k * 32 + p + 16] != (g_coef_shares[e][d][i][1][k * 16 + p].data >> 8))
                //             printf("wtf c1 hi\n");
                //         else
                //             printf("pass\n");
                //     }
                // }
            }
            random_offset += 2 * instance.lambda;

            // get sumcheck random
            h_i = phase_sumcheck_commitment(instance, salt, h_i, g_coef1_deltas[d][i]);
            std::vector<field::GF2E> rho_i = phase_sumcheck_expand(instance, h_i);
            g_rho[d].push_back(rho_i);

            // sumcheck update
            for (size_t e = 0; e < instance.num_rounds; e++) {
                auto r = rho_i[e];
                g_fr[e][d] = (g_coef1[d][i][e][1] * r + g_fr[e][d]) * r + g_coef1[d][i][e][0];
                // for (size_t p = 0; p < instance.num_MPC_parties; p++) {
                //     g_fr_shares[e][p][d] = (g_coef_shares[e][d][i][1][p] * r + g_fr_shares[e][p][d]) * r + g_coef_shares[e][d][i][0][p];
                // }
                auto r_raw = r.data;
                __m256i table0 = _mm256_loadu_si256((__m256i *)&(field::GF2E::mul_lut[r_raw][0]));
                __m256i table1 = _mm256_loadu_si256((__m256i *)&(field::GF2E::mul_lut[r_raw][8]));
                __m256i table2 = _mm256_loadu_si256((__m256i *)&(field::GF2E::mul_lut[r_raw][16]));
                __m256i table3 = _mm256_loadu_si256((__m256i *)&(field::GF2E::mul_lut[r_raw][24]));
                __m256i mask1 = _mm256_set1_epi8(0x0f);
                __m256i mask2 = _mm256_set1_epi8(0xf0);
                for (size_t k = 0; k < instance.num_MPC_parties / 16; k++) {
                    __m256i lo, hi, tmp;
                    __m256i t_coef0 = _mm256_loadu_si256((__m256i *)&(g_coef_shares_v[e][d][i][0][k * 32]));
                    __m256i t_coef1 = _mm256_loadu_si256((__m256i *)&(g_coef_shares_v[e][d][i][1][k * 32]));
                    __m256i t_fr    = _mm256_loadu_si256((__m256i *)&(g_fr_shares_v[e][d][k * 32]));

                    // coef1 * r
                    lo = _mm256_and_si256(t_coef1, mask1);  // a0, a2
                    lo = _mm256_shuffle_epi8(table0, lo);   // a0lo, a2hi
                    hi = _mm256_srli_epi64(_mm256_and_si256(t_coef1, mask2), 4);    // a1, a3
                    hi = _mm256_shuffle_epi8(table1, hi);   // a1lo, a3hi
                    tmp = _mm256_xor_si256(hi, lo);
                    t_coef1 = _mm256_permute4x64_epi64(t_coef1, 2 | (3 << 2) | (1 << 6));   // a2,a3; a0,a1
                    lo = _mm256_and_si256(t_coef1, mask1);  // a2, a0
                    lo = _mm256_shuffle_epi8(table2, lo);   // a2lo, a0hi
                    hi = _mm256_srli_epi64(_mm256_and_si256(t_coef1, mask2), 4);    // a3, a1
                    hi = _mm256_shuffle_epi8(table3, hi);   // a3lo, a1hi
                    t_coef1 = _mm256_xor_si256(tmp, _mm256_xor_si256(hi, lo));
                    // + fr
                    t_coef1 = _mm256_xor_si256(t_fr, t_coef1);
                    // * r
                    lo = _mm256_and_si256(t_coef1, mask1);  // a0, a2
                    lo = _mm256_shuffle_epi8(table0, lo);   // a0lo, a2hi
                    hi = _mm256_srli_epi64(_mm256_and_si256(t_coef1, mask2), 4);    // a1, a3
                    hi = _mm256_shuffle_epi8(table1, hi);   // a1lo, a3hi
                    tmp = _mm256_xor_si256(hi, lo);
                    t_coef1 = _mm256_permute4x64_epi64(t_coef1, 2 | (3 << 2) | (1 << 6));   // a2,a3; a0,a1
                    lo = _mm256_and_si256(t_coef1, mask1);  // a2, a0
                    lo = _mm256_shuffle_epi8(table2, lo);   // a2lo, a0hi
                    hi = _mm256_srli_epi64(_mm256_and_si256(t_coef1, mask2), 4);    // a3, a1
                    hi = _mm256_shuffle_epi8(table3, hi);   // a3lo, a1hi
                    t_coef1 = _mm256_xor_si256(tmp, _mm256_xor_si256(hi, lo));
                    // + coef[0]
                    t_coef1 = _mm256_xor_si256(t_coef0, t_coef1);

                    _mm256_storeu_si256((__m256i *)&(g_fr_shares_v[e][d][k * 32]), t_coef1);
                }

                int mask = 1 << (bit_len - i - 1);
                for (int u = 0; u < mask; u++) {
                    int u0 = u << 1;
                    int u1 = u0 + 1;
                    g_AV[e][u] = g_AV[e][u0] + g_AV[e][u1] * r;
                    g_AG[e][u] = g_AG[e][u0] + g_AG[e][u1] * r;
                }

                mask = 1 << i;
                for (int k = 0; k < mask; k++) {
                    g_Rho[e][k + mask] = g_Rho[e][k] * r;
                    g_Rho[e][k] += g_Rho[e][k + mask];
                }
            }
        }
        // end of outlayer phase 1
        for (size_t e = 0; e < instance.num_rounds; e++) {
            g_Vr[e][d] = g_AV[e][0];
        }

#ifdef TIMING
    tmp_time = timing_read(&ctx);
    printf("GGKR output layer phase 1 time: %ld\n", tmp_time - start_time);
    start_time = timing_read(&ctx);
#endif

        // output layer
        // phase 2
        // init AV, AG
        for (size_t e = 0; e < instance.num_rounds; e++) {
            g_AV[e] = gate_outputs[e][d + 1];
            g_AG[e].resize(1 << bit_len);
            for (int u = 0; u < (1 << (bit_len - 1)); u++) {
                // AG[e][2 * u + 1] = Tau[e][u] * Rho[e][2 * u]; // Tau = 1 for output layer because of only one output
                g_AG[e][2 * u + 1] = g_Rho[e][2 * u];
                // other AG should be zero
                g_AG[e][2 * u] = field::GF2E(0);
            }
        }
        g_coef2[d].resize(bit_len);
        g_coef2_deltas[d].resize(bit_len);
        for (int i = 0; i < bit_len; i++) {
            g_coef2[d][i].resize(instance.num_rounds);
            g_coef2_deltas[d][i].resize(instance.num_rounds);
            // whole computes
            for (size_t e = 0; e < instance.num_rounds; e++) {
                for (int u = 0; u < (1 << (bit_len - i)); u += 2) {
                    int one_u = u + 1;
                    g_AV[e][one_u] += g_AV[e][u];
                    g_AG[e][one_u] += g_AG[e][u];
                    g_coef2[d][i][e][0] += g_AV[e][u] * g_AG[e][u] * g_Vr[e][d];
                    g_coef2[d][i][e][1] += g_AV[e][one_u] * g_AG[e][one_u] * g_Vr[e][d];
                }
                g_fr[e][d] -= g_coef2[d][i][e][1];
            }
            // parties compute shares
            for (size_t e = 0; e < instance.num_rounds; e++) {
                g_coef2_deltas[d][i][e].resize(2);
                __m256i g_coef2_deltas_v[2];
                g_coef2_deltas_v[0] = _mm256_setzero_si256();
                g_coef2_deltas_v[1] = _mm256_setzero_si256();
                for (size_t k = 0; k < instance.num_MPC_parties / 16; k++) {
                    for (size_t p = 0; p < 16; p++) {
                        auto random_coef_share =
                            random_tapes.get_bytes(e, k * 16 + p,
                                    random_offset,
                                    2 * instance.lambda);
                        g_coef_shares_v[e][d][i][2][k * 32 + p] = *(uint8_t *)(random_coef_share.data());
                        g_coef_shares_v[e][d][i][3][k * 32 + p] = *(uint8_t *)(random_coef_share.data() + 2);
                        g_coef_shares_v[e][d][i][2][k * 32 + 16 + p] = *(uint8_t *)(random_coef_share.data() + 1);
                        g_coef_shares_v[e][d][i][3][k * 32 + 16 + p] = *(uint8_t *)(random_coef_share.data() + 3);
                    }
                    __m256i t_coef0 = _mm256_loadu_si256((__m256i *)&(g_coef_shares_v[e][d][i][2][k * 32]));
                    __m256i t_coef1 = _mm256_loadu_si256((__m256i *)&(g_coef_shares_v[e][d][i][3][k * 32]));
                    __m256i t_fr    = _mm256_loadu_si256((__m256i *)&(g_fr_shares_v[e][d][k * 32]));
                    t_fr = _mm256_xor_si256(t_coef1, t_fr);
                    _mm256_storeu_si256((__m256i *)&(g_fr_shares_v[e][d][k * 32]), t_fr);
                    g_coef2_deltas_v[0] = _mm256_xor_si256(g_coef2_deltas_v[0], t_coef0);
                    g_coef2_deltas_v[1] = _mm256_xor_si256(g_coef2_deltas_v[1], t_coef1);
                }

                g_coef2_deltas[d][i][e][0] = g_coef2[d][i][e][0] - field::GF2E(_mm256_hxor_epu16(g_coef2_deltas_v[0]));
                g_coef2_deltas[d][i][e][1] = g_coef2[d][i][e][1] - field::GF2E(_mm256_hxor_epu16(g_coef2_deltas_v[1]));

                g_coef_shares_v[e][d][i][2][0] ^= (g_coef2_deltas[d][i][e][0].data & 255);
                g_coef_shares_v[e][d][i][3][0] ^= (g_coef2_deltas[d][i][e][1].data & 255);
                g_coef_shares_v[e][d][i][2][16] ^= (g_coef2_deltas[d][i][e][0].data >> 8);
                g_coef_shares_v[e][d][i][3][16] ^= (g_coef2_deltas[d][i][e][1].data >> 8);

                g_fr_shares_v[e][d][0] ^= g_coef2_deltas[d][i][e][1].data & 255;
                g_fr_shares_v[e][d][16] ^= g_coef2_deltas[d][i][e][1].data >> 8;

                // field::GF2E delta0(0);
                // field::GF2E delta1(0);
                // for (size_t p = 0; p < instance.num_MPC_parties; p++) {
                //     auto random_coef_share =
                //         random_tapes.get_bytes(e, p,
                //                 random_offset,
                //                 2 * instance.lambda);
                //     g_coef_shares[e][d][i][2][p].from_bytes(random_coef_share.data());
                //     g_coef_shares[e][d][i][3][p].from_bytes(random_coef_share.data() + instance.lambda);
                //     g_fr_shares[e][p][d] -= g_coef_shares[e][d][i][3][p];
                //     delta0 += g_coef_shares[e][d][i][2][p];
                //     delta1 += g_coef_shares[e][d][i][3][p];
                // }
                // delta0 = g_coef2[d][i][e][0] - delta0;
                // delta1 = g_coef2[d][i][e][1] - delta1;
                // g_coef_shares[e][d][i][2][0] += delta0;
                // g_coef_shares[e][d][i][3][0] += delta1;
                // g_fr_shares[e][0][d] += delta1;

                // if (delta0 != g_coef2_deltas[d][i][e][0])
                //     printf("wtf delta0\n");
                // if (delta1 != g_coef2_deltas[d][i][e][1])
                //     printf("wtf delta1\n");
                // for (size_t k = 0; k < instance.num_MPC_parties / 16; k++) {
                //     for (int p = 0; p < 16; p++) {
                //         if (g_fr_shares_v[e][d][k * 32 + p] != (g_fr_shares[e][k * 16 + p][d].data & 255))
                //             printf("wtf fr lo\n");
                //         else if (g_fr_shares_v[e][d][k * 32 + p + 16] != (g_fr_shares[e][k * 16 + p][d].data >> 8))
                //             printf("wtf fr hi\n");
                //         else if (g_coef_shares_v[e][d][i][2][k * 32 + p] != (g_coef_shares[e][d][i][2][k * 16 + p].data & 255))
                //             printf("wtf c0 lo\n");
                //         else if (g_coef_shares_v[e][d][i][2][k * 32 + p + 16] != (g_coef_shares[e][d][i][2][k * 16 + p].data >> 8))
                //             printf("wtf c0 hi\n");
                //         else if (g_coef_shares_v[e][d][i][3][k * 32 + p] != (g_coef_shares[e][d][i][3][k * 16 + p].data & 255))
                //             printf("wtf c1 lo\n");
                //         else if (g_coef_shares_v[e][d][i][3][k * 32 + p + 16] != (g_coef_shares[e][d][i][3][k * 16 + p].data >> 8))
                //             printf("wtf c1 hi\n");
                //         else
                //             printf("pass\n");
                //     }
                // }

            }

            random_offset += 2 * instance.lambda;

            // get sumcheck random
            h_i = phase_sumcheck_commitment(instance, salt, h_i, g_coef2_deltas[d][i]);
            std::vector<field::GF2E> sgm_i = phase_sumcheck_expand(instance, h_i);
            g_sgm[d].push_back(sgm_i);

            // sumcheck update
            for (size_t e = 0; e < instance.num_rounds; e++) {
                auto r = sgm_i[e];
                g_fr[e][d] = (g_coef2[d][i][e][1] * r + g_fr[e][d]) * r + g_coef2[d][i][e][0];
                // for (size_t p = 0; p < instance.num_MPC_parties; p++) {
                //     g_fr_shares[e][p][d] = (g_coef_shares[e][d][i][3][p] * r + g_fr_shares[e][p][d]) * r + g_coef_shares[e][d][i][2][p];
                // }
                auto r_raw = r.data;
                __m256i table0 = _mm256_loadu_si256((__m256i *)&(field::GF2E::mul_lut[r_raw][0]));
                __m256i table1 = _mm256_loadu_si256((__m256i *)&(field::GF2E::mul_lut[r_raw][8]));
                __m256i table2 = _mm256_loadu_si256((__m256i *)&(field::GF2E::mul_lut[r_raw][16]));
                __m256i table3 = _mm256_loadu_si256((__m256i *)&(field::GF2E::mul_lut[r_raw][24]));
                __m256i mask1 = _mm256_set1_epi8(0x0f);
                __m256i mask2 = _mm256_set1_epi8(0xf0);
                for (size_t k = 0; k < instance.num_MPC_parties / 16; k++) {
                    __m256i lo, hi, tmp;
                    __m256i t_coef0 = _mm256_loadu_si256((__m256i *)&(g_coef_shares_v[e][d][i][2][k * 32]));
                    __m256i t_coef1 = _mm256_loadu_si256((__m256i *)&(g_coef_shares_v[e][d][i][3][k * 32]));
                    __m256i t_fr    = _mm256_loadu_si256((__m256i *)&(g_fr_shares_v[e][d][k * 32]));

                    // coef1 * r
                    lo = _mm256_and_si256(t_coef1, mask1);  // a0, a2
                    lo = _mm256_shuffle_epi8(table0, lo);   // a0lo, a2hi
                    hi = _mm256_srli_epi64(_mm256_and_si256(t_coef1, mask2), 4);    // a1, a3
                    hi = _mm256_shuffle_epi8(table1, hi);   // a1lo, a3hi
                    tmp = _mm256_xor_si256(hi, lo);
                    t_coef1 = _mm256_permute4x64_epi64(t_coef1, 2 | (3 << 2) | (1 << 6));   // a2,a3; a0,a1
                    lo = _mm256_and_si256(t_coef1, mask1);  // a2, a0
                    lo = _mm256_shuffle_epi8(table2, lo);   // a2lo, a0hi
                    hi = _mm256_srli_epi64(_mm256_and_si256(t_coef1, mask2), 4);    // a3, a1
                    hi = _mm256_shuffle_epi8(table3, hi);   // a3lo, a1hi
                    t_coef1 = _mm256_xor_si256(tmp, _mm256_xor_si256(hi, lo));
                    // + fr
                    t_coef1 = _mm256_xor_si256(t_fr, t_coef1);
                    // * r
                    lo = _mm256_and_si256(t_coef1, mask1);  // a0, a2
                    lo = _mm256_shuffle_epi8(table0, lo);   // a0lo, a2hi
                    hi = _mm256_srli_epi64(_mm256_and_si256(t_coef1, mask2), 4);    // a1, a3
                    hi = _mm256_shuffle_epi8(table1, hi);   // a1lo, a3hi
                    tmp = _mm256_xor_si256(hi, lo);
                    t_coef1 = _mm256_permute4x64_epi64(t_coef1, 2 | (3 << 2) | (1 << 6));   // a2,a3; a0,a1
                    lo = _mm256_and_si256(t_coef1, mask1);  // a2, a0
                    lo = _mm256_shuffle_epi8(table2, lo);   // a2lo, a0hi
                    hi = _mm256_srli_epi64(_mm256_and_si256(t_coef1, mask2), 4);    // a3, a1
                    hi = _mm256_shuffle_epi8(table3, hi);   // a3lo, a1hi
                    t_coef1 = _mm256_xor_si256(tmp, _mm256_xor_si256(hi, lo));
                    // + coef[0]
                    t_coef1 = _mm256_xor_si256(t_coef0, t_coef1);

                    _mm256_storeu_si256((__m256i *)&(g_fr_shares_v[e][d][k * 32]), t_coef1);
                }

                int mask = 1 << (bit_len - i - 1);
                for (int u = 0; u < mask; u++) {
                    int u0 = u << 1;
                    int u1 = u0 + 1;
                    g_AV[e][u] = g_AV[e][u0] + g_AV[e][u1] * r;
                    g_AG[e][u] = g_AG[e][u0] + g_AG[e][u1] * r;
                }

                mask = 1 << i;
                for (int k = 0; k < mask; k++) {
                    g_Sgm[e][k + mask] = g_Sgm[e][k] * r;
                    g_Sgm[e][k] += g_Sgm[e][k + mask];
                }

                // update pred
                field::GF2E one(1);
                // if (i == 0)
                    g_pred[e][d] *= (one + g_rho[d][i][e]) * g_sgm[d][i][e];
                // else
                //     g_pred[e][d] *= g_rho[d][i][e] + g_sgm[d][i][e] + one;
            }
        }
        // end of outlayer phase 2
        for (size_t e = 0; e < instance.num_rounds; e++) {
            g_Vs[e][d] = g_AV[e][0];
        }
        // sanity check
        // for (size_t e = 0; e < instance.num_rounds; e++) {
        //     if (g_fr[e][d] != g_Vr[e][d] * g_Vs[e][d] * g_pred[e][d]) {
        //         throw std::runtime_error("sanity check group gkr output layer failed.");
        //     }
        // }

#ifdef TIMING
    tmp_time = timing_read(&ctx);
    printf("GGKR output layer phase 2 time: %ld\n", tmp_time - start_time);
    start_time = timing_read(&ctx);
#endif

        // loop over layers
        for (d = 1; d < depth; d++) {
            // commit Vr, Vs
            g_Vr_deltas[d-1].resize(instance.num_rounds);
            g_Vs_deltas[d-1].resize(instance.num_rounds);
            for (size_t e = 0; e < instance.num_rounds; e++) {
                __m256i Vr_deltas_v = _mm256_setzero_si256();
                __m256i Vs_deltas_v = _mm256_setzero_si256();
                for (size_t k = 0; k < instance.num_MPC_parties / 16; k++) {
                    for (size_t p = 0; p < 16; p++) {
                        auto random_coef_share =
                            random_tapes.get_bytes(e, k * 16 + p,
                                    random_offset,
                                    2 * instance.lambda);
                        g_Vr_shares_v[e][d-1][k * 32 + p] = *(uint8_t *)(random_coef_share.data());
                        g_Vs_shares_v[e][d-1][k * 32 + p] = *(uint8_t *)(random_coef_share.data() + 2);
                        g_Vr_shares_v[e][d-1][k * 32 + 16 + p] = *(uint8_t *)(random_coef_share.data() + 1);
                        g_Vs_shares_v[e][d-1][k * 32 + 16 + p] = *(uint8_t *)(random_coef_share.data() + 3);
                    }
                    __m256i t_coef0 = _mm256_loadu_si256((__m256i *)&(g_Vr_shares_v[e][d-1][k * 32]));
                    __m256i t_coef1 = _mm256_loadu_si256((__m256i *)&(g_Vs_shares_v[e][d-1][k * 32]));
                    Vr_deltas_v = _mm256_xor_si256(Vr_deltas_v, t_coef0);
                    Vs_deltas_v = _mm256_xor_si256(Vs_deltas_v, t_coef1);
                }
                g_Vr_deltas[d-1][e] = g_Vr[e][d-1] - field::GF2E(_mm256_hxor_epu16(Vr_deltas_v));
                g_Vs_deltas[d-1][e] = g_Vs[e][d-1] - field::GF2E(_mm256_hxor_epu16(Vs_deltas_v));
                g_Vr_shares_v[e][d-1][0]  ^= (g_Vr_deltas[d-1][e].data & 255);
                g_Vs_shares_v[e][d-1][0]  ^= (g_Vs_deltas[d-1][e].data & 255);
                g_Vr_shares_v[e][d-1][16] ^= (g_Vr_deltas[d-1][e].data >> 8);
                g_Vs_shares_v[e][d-1][16] ^= (g_Vs_deltas[d-1][e].data >> 8);

                // field::GF2E delta0(0);
                // field::GF2E delta1(0);
                // for (size_t p = 0; p < instance.num_MPC_parties; p++) {
                //     auto random_coef_share =
                //         random_tapes.get_bytes(e, p,
                //                 random_offset,
                //                 2 * instance.lambda);
                //     g_Vr_shares[e][p][d-1].from_bytes(random_coef_share.data());
                //     g_Vs_shares[e][p][d-1].from_bytes(random_coef_share.data() + instance.lambda);
                //     delta0 += g_Vr_shares[e][p][d-1];
                //     delta1 += g_Vs_shares[e][p][d-1];
                // }
                // delta0 = g_Vr[e][d-1] - delta0;
                // delta1 = g_Vs[e][d-1] - delta1;
                // g_Vr_shares[e][0][d-1] += delta0;
                // g_Vs_shares[e][0][d-1] += delta1;

                // printf("layer %d\n", d);
                // if (delta0 != g_Vr_deltas[d-1][e])
                //     printf("wtf delta0\n");
                // if (delta1 != g_Vs_deltas[d-1][e])
                //     printf("wtf delta1\n");
                // for (size_t k = 0; k < instance.num_MPC_parties / 16; k++) {
                //     for (int p = 0; p < 16; p++) {
                //         if (g_Vr_shares_v[e][d-1][k * 32 + p] != (g_Vr_shares[e][k * 16 + p][d-1].data & 255))
                //             printf("wtf fr lo\n");
                //         else if (g_Vr_shares_v[e][d-1][k * 32 + p + 16] != (g_Vr_shares[e][k * 16 + p][d-1].data >> 8))
                //             printf("wtf fr hi\n");
                //         else if (g_Vs_shares_v[e][d-1][k * 32 + p] != (g_Vs_shares[e][k * 16 + p][d-1].data & 255))
                //             printf("wtf fr lo\n");
                //         else if (g_Vs_shares_v[e][d-1][k * 32 + p + 16] != (g_Vs_shares[e][k * 16 + p][d-1].data >> 8))
                //             printf("wtf fr hi\n");
                //         else
                //             printf("pass\n");
                //     }
                // }
                // if (delta0 != g_coef1_deltas[d][i][e][0])
                //     printf("wtf delta0\n");
                // if (delta1 != g_coef1_deltas[d][i][e][1])
                //     printf("wtf delta1\n");
                // for (size_t k = 0; k < instance.num_MPC_parties / 16; k++) {
                //     for (int p = 0; p < 16; p++) {
                //         if (g_fr_shares_v[e][d][k * 32 + p] != (g_fr_shares[e][k * 16 + p][d].data & 255))
                //             printf("wtf fr lo\n");
                //         else if (g_fr_shares_v[e][d][k * 32 + p + 16] != (g_fr_shares[e][k * 16 + p][d].data >> 8))
                //             printf("wtf fr hi\n");
                //         else if (g_coef_shares_v[e][d][i][0][k * 32 + p] != (g_coef_shares[e][d][i][0][k * 16 + p].data & 255))
                //             printf("wtf c0 lo\n");
                //         else if (g_coef_shares_v[e][d][i][0][k * 32 + p + 16] != (g_coef_shares[e][d][i][0][k * 16 + p].data >> 8))
                //             printf("wtf c0 hi\n");
                //         else if (g_coef_shares_v[e][d][i][1][k * 32 + p] != (g_coef_shares[e][d][i][1][k * 16 + p].data & 255))
                //             printf("wtf c1 lo\n");
                //         else if (g_coef_shares_v[e][d][i][1][k * 32 + p + 16] != (g_coef_shares[e][d][i][1][k * 16 + p].data >> 8))
                //             printf("wtf c1 hi\n");
                //         else
                //             printf("pass\n");
                //     }
                // }
            }
            random_offset += 2 * instance.lambda;
            // generate random mu, nu
            h_i = phase_sumcheck_final_commitment(instance, salt, h_i, g_Vr_deltas[d-1], g_Vs_deltas[d-1]);
            std::vector<std::vector<field::GF2E>> scalar_d = phase_sumcheck_final_expand(instance, h_i);
            g_scalar[d-1] = scalar_d;
            // fr sum and shares for the next layer
            for (size_t e = 0; e < instance.num_rounds; e++) {
                g_fr[e][d] = scalar_d[e][0] * g_Vr[e][d-1] + scalar_d[e][1] * g_Vs[e][d-1];
                // for (size_t p = 0; p < instance.num_MPC_parties; p++) {
                //     g_fr_shares[e][p][d] = scalar_d[e][0] * g_Vr_shares[e][p][d-1] + scalar_d[e][1] * g_Vs_shares[e][p][d-1];
                // }
                const __m256i table00 = _mm256_loadu_si256((__m256i *)&(field::GF2E::mul_lut[scalar_d[e][0].data][0]));
                const __m256i table10 = _mm256_loadu_si256((__m256i *)&(field::GF2E::mul_lut[scalar_d[e][0].data][8]));
                const __m256i table20 = _mm256_loadu_si256((__m256i *)&(field::GF2E::mul_lut[scalar_d[e][0].data][16]));
                const __m256i table30 = _mm256_loadu_si256((__m256i *)&(field::GF2E::mul_lut[scalar_d[e][0].data][24]));
                const __m256i table01 = _mm256_loadu_si256((__m256i *)&(field::GF2E::mul_lut[scalar_d[e][1].data][0]));
                const __m256i table11 = _mm256_loadu_si256((__m256i *)&(field::GF2E::mul_lut[scalar_d[e][1].data][8]));
                const __m256i table21 = _mm256_loadu_si256((__m256i *)&(field::GF2E::mul_lut[scalar_d[e][1].data][16]));
                const __m256i table31 = _mm256_loadu_si256((__m256i *)&(field::GF2E::mul_lut[scalar_d[e][1].data][24]));
                const __m256i mask1 = _mm256_set1_epi8(0x0f);
                const __m256i mask2 = _mm256_set1_epi8(0xf0);
                for (size_t k = 0; k < instance.num_MPC_parties / 16; k++) {
                    __m256i lo, hi, tmp;
                    __m256i t_Vr = _mm256_loadu_si256((__m256i *)&(g_Vr_shares_v[e][d-1][k * 32]));
                    __m256i t_Vs = _mm256_loadu_si256((__m256i *)&(g_Vs_shares_v[e][d-1][k * 32]));

                    // t_Vr * scalar[0]
                    lo = _mm256_and_si256(t_Vr, mask1);  // a0, a2
                    lo = _mm256_shuffle_epi8(table00, lo);   // a0lo, a2hi
                    hi = _mm256_srli_epi64(_mm256_and_si256(t_Vr, mask2), 4);    // a1, a3
                    hi = _mm256_shuffle_epi8(table10, hi);   // a1lo, a3hi
                    tmp = _mm256_xor_si256(hi, lo);
                    t_Vr = _mm256_permute4x64_epi64(t_Vr, 2 | (3 << 2) | (1 << 6));   // a2,a3; a0,a1
                    lo = _mm256_and_si256(t_Vr, mask1);  // a2, a0
                    lo = _mm256_shuffle_epi8(table20, lo);   // a2lo, a0hi
                    hi = _mm256_srli_epi64(_mm256_and_si256(t_Vr, mask2), 4);    // a3, a1
                    hi = _mm256_shuffle_epi8(table30, hi);   // a3lo, a1hi
                    t_Vr = _mm256_xor_si256(tmp, _mm256_xor_si256(hi, lo));
                    // t_Vs * scalar[1]
                    lo = _mm256_and_si256(t_Vs, mask1);  // a0, a2
                    lo = _mm256_shuffle_epi8(table01, lo);   // a0lo, a2hi
                    hi = _mm256_srli_epi64(_mm256_and_si256(t_Vs, mask2), 4);    // a1, a3
                    hi = _mm256_shuffle_epi8(table11, hi);   // a1lo, a3hi
                    tmp = _mm256_xor_si256(hi, lo);
                    t_Vs = _mm256_permute4x64_epi64(t_Vs, 2 | (3 << 2) | (1 << 6));   // a2,a3; a0,a1
                    lo = _mm256_and_si256(t_Vs, mask1);  // a2, a0
                    lo = _mm256_shuffle_epi8(table21, lo);   // a2lo, a0hi
                    hi = _mm256_srli_epi64(_mm256_and_si256(t_Vs, mask2), 4);    // a3, a1
                    hi = _mm256_shuffle_epi8(table31, hi);   // a3lo, a1hi
                    t_Vs = _mm256_xor_si256(tmp, _mm256_xor_si256(hi, lo));
                    // +
                    _mm256_storeu_si256((__m256i *)&(g_fr_shares_v[e][d][k * 32]), _mm256_xor_si256(t_Vr, t_Vs));
                    // for (int p = 0; p < 16; p++) {
                    //     if (g_fr_shares_v[e][d][k * 32 + p] != (g_fr_shares[e][k * 16 + p][d].data & 255))
                    //         printf("wtf0\n");
                    //     else if (g_fr_shares_v[e][d][k * 32 + p + 16] != (g_fr_shares[e][k * 16 + p][d].data >> 8))
                    //         printf("wtf1\n");
                    //     else
                    //         printf("pass\n");
                    // }
                }
            }

            bit_len++;
            g_Rho_old = g_Rho;
            g_Sgm_old = g_Sgm;
            // init AV, AG, Rho, Sgm
            for (size_t e = 0; e < instance.num_rounds; e++) {
                g_AV[e] = gate_outputs[e][d + 1];
                g_AG[e].resize(1 << bit_len);
                g_Rho[e].resize(1 << (d + 1));
                g_Sgm[e].resize(1 << (d + 1));
                g_Rho[e][0] = field::GF2E(1);
                g_Sgm[e][0] = field::GF2E(1);
                for (int u = 0; u < (1 << (bit_len - 1)); u++) {
                    g_AG[e][2 * u] = (scalar_d[e][0] * g_Rho_old[e][u] + scalar_d[e][1] * g_Sgm_old[e][u]) * g_AV[e][2 * u + 1];
                    g_AG[e][2 * u + 1] = field::GF2E(0);
                }
            }
            // compute coefs
            g_coef1[d].resize(bit_len);
            g_coef1_deltas[d].resize(bit_len);
            for (int i = 0; i < bit_len; i++) {
                g_coef1[d][i].resize(instance.num_rounds);
                g_coef1_deltas[d][i].resize(instance.num_rounds);
                // whole computes
                for (size_t e = 0; e < instance.num_rounds; e++) {
                    // field::GF2E tmp(0);
                    for (int u = 0; u < (1 << (bit_len - i)); u += 2) {
                        int one_u = u + 1;
                        g_AV[e][one_u] += g_AV[e][u];
                        g_AG[e][one_u] += g_AG[e][u];
                        g_coef1[d][i][e][0] += g_AV[e][u] * g_AG[e][u];
                        // tmp += (g_AV[e][u] * g_AG[e][one_u] + g_AG[e][u] * g_AV[e][one_u]);
                        g_coef1[d][i][e][1] += g_AV[e][one_u] * g_AG[e][one_u];
                    }
                    g_fr[e][d] -= g_coef1[d][i][e][1];
                    // if (tmp != g_fr[e][d])
                    //     printf("wtffffffffffff %d\n", i);
                }
                // parties compute shares
                for (size_t e = 0; e < instance.num_rounds; e++) {
                    g_coef1_deltas[d][i][e].resize(2);
                    __m256i g_coef1_deltas_v[2];
                    g_coef1_deltas_v[0] = _mm256_setzero_si256();
                    g_coef1_deltas_v[1] = _mm256_setzero_si256();

                    for (size_t k = 0; k < instance.num_MPC_parties / 16; k++) {
                        for (size_t p = 0; p < 16; p++) {
                            auto random_coef_share =
                                random_tapes.get_bytes(e, k * 16 + p,
                                        random_offset,
                                        2 * instance.lambda);
                            g_coef_shares_v[e][d][i][0][k * 32 + p] = *(uint8_t *)(random_coef_share.data());
                            g_coef_shares_v[e][d][i][1][k * 32 + p] = *(uint8_t *)(random_coef_share.data() + 2);
                            g_coef_shares_v[e][d][i][0][k * 32 + 16 + p] = *(uint8_t *)(random_coef_share.data() + 1);
                            g_coef_shares_v[e][d][i][1][k * 32 + 16 + p] = *(uint8_t *)(random_coef_share.data() + 3);
                        }
                        __m256i t_coef0 = _mm256_loadu_si256((__m256i *)&(g_coef_shares_v[e][d][i][0][k * 32]));
                        __m256i t_coef1 = _mm256_loadu_si256((__m256i *)&(g_coef_shares_v[e][d][i][1][k * 32]));
                        __m256i t_fr    = _mm256_loadu_si256((__m256i *)&(g_fr_shares_v[e][d][k * 32]));
                        t_fr = _mm256_xor_si256(t_coef1, t_fr);
                        _mm256_storeu_si256((__m256i *)&(g_fr_shares_v[e][d][k * 32]), t_fr);
                        g_coef1_deltas_v[0] = _mm256_xor_si256(g_coef1_deltas_v[0], t_coef0);
                        g_coef1_deltas_v[1] = _mm256_xor_si256(g_coef1_deltas_v[1], t_coef1);
                    }

                    g_coef1_deltas[d][i][e][0] = g_coef1[d][i][e][0] - field::GF2E(_mm256_hxor_epu16(g_coef1_deltas_v[0]));
                    g_coef1_deltas[d][i][e][1] = g_coef1[d][i][e][1] - field::GF2E(_mm256_hxor_epu16(g_coef1_deltas_v[1]));

                    g_coef_shares_v[e][d][i][0][0] ^= (g_coef1_deltas[d][i][e][0].data & 255);
                    g_coef_shares_v[e][d][i][1][0] ^= (g_coef1_deltas[d][i][e][1].data & 255);
                    g_coef_shares_v[e][d][i][0][16] ^= (g_coef1_deltas[d][i][e][0].data >> 8);
                    g_coef_shares_v[e][d][i][1][16] ^= (g_coef1_deltas[d][i][e][1].data >> 8);

                    g_fr_shares_v[e][d][0] ^= g_coef1_deltas[d][i][e][1].data & 255;
                    g_fr_shares_v[e][d][16] ^= g_coef1_deltas[d][i][e][1].data >> 8;

                    // field::GF2E delta0(0);
                    // field::GF2E delta1(0);
                    // for (size_t p = 0; p < instance.num_MPC_parties; p++) {
                    //     auto random_coef_share =
                    //         random_tapes.get_bytes(e, p,
                    //                 random_offset,
                    //                 2 * instance.lambda);
                    //     g_coef_shares[e][d][i][0][p].from_bytes(random_coef_share.data());
                    //     g_coef_shares[e][d][i][1][p].from_bytes(random_coef_share.data() + instance.lambda);
                    //     g_fr_shares[e][p][d] -= g_coef_shares[e][d][i][1][p];
                    //     delta0 += g_coef_shares[e][d][i][0][p];
                    //     delta1 += g_coef_shares[e][d][i][1][p];
                    // }
                    // delta0 = g_coef1[d][i][e][0] - delta0;
                    // delta1 = g_coef1[d][i][e][1] - delta1;
                    // g_coef_shares[e][d][i][0][0] += delta0;
                    // g_coef_shares[e][d][i][1][0] += delta1;
                    // g_fr_shares[e][0][d] += delta1;

                    // printf("layer %d phase 1 step %d\n", d, i);
                    // if (delta0 != g_coef1_deltas[d][i][e][0])
                    //     printf("wtf delta0\n");
                    // if (delta1 != g_coef1_deltas[d][i][e][1])
                    //     printf("wtf delta1\n");
                    // for (size_t k = 0; k < instance.num_MPC_parties / 16; k++) {
                    //     for (int p = 0; p < 16; p++) {
                    //         if (g_fr_shares_v[e][d][k * 32 + p] != (g_fr_shares[e][k * 16 + p][d].data & 255))
                    //             printf("wtf fr lo\n");
                    //         else if (g_fr_shares_v[e][d][k * 32 + p + 16] != (g_fr_shares[e][k * 16 + p][d].data >> 8))
                    //             printf("wtf fr hi\n");
                    //         else if (g_coef_shares_v[e][d][i][0][k * 32 + p] != (g_coef_shares[e][d][i][0][k * 16 + p].data & 255))
                    //             printf("wtf c0 lo\n");
                    //         else if (g_coef_shares_v[e][d][i][0][k * 32 + p + 16] != (g_coef_shares[e][d][i][0][k * 16 + p].data >> 8))
                    //             printf("wtf c0 hi\n");
                    //         else if (g_coef_shares_v[e][d][i][1][k * 32 + p] != (g_coef_shares[e][d][i][1][k * 16 + p].data & 255))
                    //             printf("wtf c1 lo\n");
                    //         else if (g_coef_shares_v[e][d][i][1][k * 32 + p + 16] != (g_coef_shares[e][d][i][1][k * 16 + p].data >> 8))
                    //             printf("wtf c1 hi\n");
                    //         else
                    //             printf("pass\n");
                    //     }
                    // }
                }

                random_offset += 2 * instance.lambda;

                // get sumcheck random
                h_i = phase_sumcheck_commitment(instance, salt, h_i, g_coef1_deltas[d][i]);
                std::vector<field::GF2E> rho_i = phase_sumcheck_expand(instance, h_i);
                g_rho[d].push_back(rho_i);

                // sumcheck update
                for (size_t e = 0; e < instance.num_rounds; e++) {
                    auto r = rho_i[e];
                    g_fr[e][d] = (g_coef1[d][i][e][1] * r + g_fr[e][d]) * r + g_coef1[d][i][e][0];
                    // for (size_t p = 0; p < instance.num_MPC_parties; p++) {
                    //     g_fr_shares[e][p][d] = (g_coef_shares[e][d][i][1][p] * r + g_fr_shares[e][p][d]) * r + g_coef_shares[e][d][i][0][p];
                    // }
                    auto r_raw = r.data;
                    __m256i table0 = _mm256_loadu_si256((__m256i *)&(field::GF2E::mul_lut[r_raw][0]));
                    __m256i table1 = _mm256_loadu_si256((__m256i *)&(field::GF2E::mul_lut[r_raw][8]));
                    __m256i table2 = _mm256_loadu_si256((__m256i *)&(field::GF2E::mul_lut[r_raw][16]));
                    __m256i table3 = _mm256_loadu_si256((__m256i *)&(field::GF2E::mul_lut[r_raw][24]));
                    __m256i mask1 = _mm256_set1_epi8(0x0f);
                    __m256i mask2 = _mm256_set1_epi8(0xf0);
                    for (size_t k = 0; k < instance.num_MPC_parties / 16; k++) {
                        __m256i lo, hi, tmp;
                        __m256i t_coef0 = _mm256_loadu_si256((__m256i *)&(g_coef_shares_v[e][d][i][0][k * 32]));
                        __m256i t_coef1 = _mm256_loadu_si256((__m256i *)&(g_coef_shares_v[e][d][i][1][k * 32]));
                        __m256i t_fr    = _mm256_loadu_si256((__m256i *)&(g_fr_shares_v[e][d][k * 32]));

                        // coef1 * r
                        lo = _mm256_and_si256(t_coef1, mask1);  // a0, a2
                        lo = _mm256_shuffle_epi8(table0, lo);   // a0lo, a2hi
                        hi = _mm256_srli_epi64(_mm256_and_si256(t_coef1, mask2), 4);    // a1, a3
                        hi = _mm256_shuffle_epi8(table1, hi);   // a1lo, a3hi
                        tmp = _mm256_xor_si256(hi, lo);
                        t_coef1 = _mm256_permute4x64_epi64(t_coef1, 2 | (3 << 2) | (1 << 6));   // a2,a3; a0,a1
                        lo = _mm256_and_si256(t_coef1, mask1);  // a2, a0
                        lo = _mm256_shuffle_epi8(table2, lo);   // a2lo, a0hi
                        hi = _mm256_srli_epi64(_mm256_and_si256(t_coef1, mask2), 4);    // a3, a1
                        hi = _mm256_shuffle_epi8(table3, hi);   // a3lo, a1hi
                        t_coef1 = _mm256_xor_si256(tmp, _mm256_xor_si256(hi, lo));
                        // + fr
                        t_coef1 = _mm256_xor_si256(t_fr, t_coef1);
                        // * r
                        lo = _mm256_and_si256(t_coef1, mask1);  // a0, a2
                        lo = _mm256_shuffle_epi8(table0, lo);   // a0lo, a2hi
                        hi = _mm256_srli_epi64(_mm256_and_si256(t_coef1, mask2), 4);    // a1, a3
                        hi = _mm256_shuffle_epi8(table1, hi);   // a1lo, a3hi
                        tmp = _mm256_xor_si256(hi, lo);
                        t_coef1 = _mm256_permute4x64_epi64(t_coef1, 2 | (3 << 2) | (1 << 6));   // a2,a3; a0,a1
                        lo = _mm256_and_si256(t_coef1, mask1);  // a2, a0
                        lo = _mm256_shuffle_epi8(table2, lo);   // a2lo, a0hi
                        hi = _mm256_srli_epi64(_mm256_and_si256(t_coef1, mask2), 4);    // a3, a1
                        hi = _mm256_shuffle_epi8(table3, hi);   // a3lo, a1hi
                        t_coef1 = _mm256_xor_si256(tmp, _mm256_xor_si256(hi, lo));
                        // + coef[0]
                        t_coef1 = _mm256_xor_si256(t_coef0, t_coef1);

                        _mm256_storeu_si256((__m256i *)&(g_fr_shares_v[e][d][k * 32]), t_coef1);
                    }

                    int mask = 1 << (bit_len - i - 1);
                    for (int u = 0; u < mask; u++) {
                        int u0 = u << 1;
                        int u1 = u0 + 1;
                        g_AV[e][u] = g_AV[e][u0] + g_AV[e][u1] * r;
                        g_AG[e][u] = g_AG[e][u0] + g_AG[e][u1] * r;
                    }

                    mask = 1 << i;
                    for (int k = 0; k < mask; k++) {
                        g_Rho[e][k + mask] = g_Rho[e][k] * r;
                        g_Rho[e][k] += g_Rho[e][k + mask];
                    }
                }
            }
            // end of outlayer phase 1
            for (size_t e = 0; e < instance.num_rounds; e++) {
                g_Vr[e][d] = g_AV[e][0];
            }

            // phase 2
            // init AV, AG
            for (size_t e = 0; e < instance.num_rounds; e++) {
                g_AV[e] = gate_outputs[e][d + 1];
                g_AG[e].resize(1 << bit_len);
                for (int u = 0; u < (1 << (bit_len - 1)); u++) {
                    g_AG[e][2 * u + 1] = (scalar_d[e][0] * g_Rho_old[e][u] + scalar_d[e][1] * g_Sgm_old[e][u]) * g_Rho[e][2 * u];
                    g_AG[e][2 * u] = field::GF2E(0);
                }
            }
            g_coef2[d].resize(bit_len);
            g_coef2_deltas[d].resize(bit_len);
            for (int i = 0; i < bit_len; i++) {
                g_coef2[d][i].resize(instance.num_rounds);
                g_coef2_deltas[d][i].resize(instance.num_rounds);
                // whole computes
                for (size_t e = 0; e < instance.num_rounds; e++) {
                    // field::GF2E tmp(0);
                    for (int u = 0; u < (1 << (bit_len - i)); u += 2) {
                        int one_u = u + 1;
                        g_AV[e][one_u] += g_AV[e][u];
                        g_AG[e][one_u] += g_AG[e][u];
                        g_coef2[d][i][e][0] += g_AV[e][u] * g_AG[e][u] * g_Vr[e][d];
                        // tmp += (g_AV[e][u] * g_AG[e][one_u] + g_AG[e][u] * g_AV[e][one_u]) * g_Vr[e][d];
                        g_coef2[d][i][e][1] += g_AV[e][one_u] * g_AG[e][one_u] * g_Vr[e][d];
                    }
                    g_fr[e][d] -= g_coef2[d][i][e][1];
                    // if (tmp != g_fr[e][d])
                    //     printf("wtffffffffffff %d\n", i);
                }
                // parties compute shares
                for (size_t e = 0; e < instance.num_rounds; e++) {
                    g_coef2_deltas[d][i][e].resize(2);
                    __m256i g_coef2_deltas_v[2];
                    g_coef2_deltas_v[0] = _mm256_setzero_si256();
                    g_coef2_deltas_v[1] = _mm256_setzero_si256();
                    for (size_t k = 0; k < instance.num_MPC_parties / 16; k++) {
                        for (size_t p = 0; p < 16; p++) {
                            auto random_coef_share =
                                random_tapes.get_bytes(e, k * 16 + p,
                                        random_offset,
                                        2 * instance.lambda);
                            g_coef_shares_v[e][d][i][2][k * 32 + p] = *(uint8_t *)(random_coef_share.data());
                            g_coef_shares_v[e][d][i][3][k * 32 + p] = *(uint8_t *)(random_coef_share.data() + 2);
                            g_coef_shares_v[e][d][i][2][k * 32 + 16 + p] = *(uint8_t *)(random_coef_share.data() + 1);
                            g_coef_shares_v[e][d][i][3][k * 32 + 16 + p] = *(uint8_t *)(random_coef_share.data() + 3);
                        }
                        __m256i t_coef0 = _mm256_loadu_si256((__m256i *)&(g_coef_shares_v[e][d][i][2][k * 32]));
                        __m256i t_coef1 = _mm256_loadu_si256((__m256i *)&(g_coef_shares_v[e][d][i][3][k * 32]));
                        __m256i t_fr    = _mm256_loadu_si256((__m256i *)&(g_fr_shares_v[e][d][k * 32]));
                        t_fr = _mm256_xor_si256(t_coef1, t_fr);
                        _mm256_storeu_si256((__m256i *)&(g_fr_shares_v[e][d][k * 32]), t_fr);
                        g_coef2_deltas_v[0] = _mm256_xor_si256(g_coef2_deltas_v[0], t_coef0);
                        g_coef2_deltas_v[1] = _mm256_xor_si256(g_coef2_deltas_v[1], t_coef1);
                    }

                    g_coef2_deltas[d][i][e][0] = g_coef2[d][i][e][0] - field::GF2E(_mm256_hxor_epu16(g_coef2_deltas_v[0]));
                    g_coef2_deltas[d][i][e][1] = g_coef2[d][i][e][1] - field::GF2E(_mm256_hxor_epu16(g_coef2_deltas_v[1]));

                    g_coef_shares_v[e][d][i][2][0] ^= (g_coef2_deltas[d][i][e][0].data & 255);
                    g_coef_shares_v[e][d][i][3][0] ^= (g_coef2_deltas[d][i][e][1].data & 255);
                    g_coef_shares_v[e][d][i][2][16] ^= (g_coef2_deltas[d][i][e][0].data >> 8);
                    g_coef_shares_v[e][d][i][3][16] ^= (g_coef2_deltas[d][i][e][1].data >> 8);

                    g_fr_shares_v[e][d][0] ^= g_coef2_deltas[d][i][e][1].data & 255;
                    g_fr_shares_v[e][d][16] ^= g_coef2_deltas[d][i][e][1].data >> 8;

                    // field::GF2E delta0(0);
                    // field::GF2E delta1(0);
                    // for (size_t p = 0; p < instance.num_MPC_parties; p++) {
                    //     auto random_coef_share =
                    //         random_tapes.get_bytes(e, p,
                    //                 random_offset,
                    //                 2 * instance.lambda);
                    //     g_coef_shares[e][d][i][2][p].from_bytes(random_coef_share.data());
                    //     g_coef_shares[e][d][i][3][p].from_bytes(random_coef_share.data() + instance.lambda);
                    //     g_fr_shares[e][p][d] -= g_coef_shares[e][d][i][3][p];
                    //     delta0 += g_coef_shares[e][d][i][2][p];
                    //     delta1 += g_coef_shares[e][d][i][3][p];
                    // }
                    // delta0 = g_coef2[d][i][e][0] - delta0;
                    // delta1 = g_coef2[d][i][e][1] - delta1;
                    // g_coef_shares[e][d][i][2][0] += delta0;
                    // g_coef_shares[e][d][i][3][0] += delta1;
                    // g_fr_shares[e][0][d] += delta1;

                    // printf("layer %d phase 2 step %d\n", d, i);
                    // if (delta0 != g_coef2_deltas[d][i][e][0])
                    //     printf("wtf delta0\n");
                    // if (delta1 != g_coef2_deltas[d][i][e][1])
                    //     printf("wtf delta1\n");
                    // for (size_t k = 0; k < instance.num_MPC_parties / 16; k++) {
                    //     for (int p = 0; p < 16; p++) {
                    //         if (g_fr_shares_v[e][d][k * 32 + p] != (g_fr_shares[e][k * 16 + p][d].data & 255))
                    //             printf("wtf fr lo\n");
                    //         else if (g_fr_shares_v[e][d][k * 32 + p + 16] != (g_fr_shares[e][k * 16 + p][d].data >> 8))
                    //             printf("wtf fr hi\n");
                    //         else if (g_coef_shares_v[e][d][i][2][k * 32 + p] != (g_coef_shares[e][d][i][2][k * 16 + p].data & 255))
                    //             printf("wtf c0 lo\n");
                    //         else if (g_coef_shares_v[e][d][i][2][k * 32 + p + 16] != (g_coef_shares[e][d][i][2][k * 16 + p].data >> 8))
                    //             printf("wtf c0 hi\n");
                    //         else if (g_coef_shares_v[e][d][i][3][k * 32 + p] != (g_coef_shares[e][d][i][3][k * 16 + p].data & 255))
                    //             printf("wtf c1 lo\n");
                    //         else if (g_coef_shares_v[e][d][i][3][k * 32 + p + 16] != (g_coef_shares[e][d][i][3][k * 16 + p].data >> 8))
                    //             printf("wtf c1 hi\n");
                    //         else
                    //             printf("pass\n");
                    //     }
                    // }
                }
                random_offset += 2 * instance.lambda;

                // get sumcheck random
                h_i = phase_sumcheck_commitment(instance, salt, h_i, g_coef2_deltas[d][i]);
                std::vector<field::GF2E> sgm_i = phase_sumcheck_expand(instance, h_i);
                g_sgm[d].push_back(sgm_i);

                // sumcheck update
                for (size_t e = 0; e < instance.num_rounds; e++) {
                    auto r = sgm_i[e];
                    g_fr[e][d] = (g_coef2[d][i][e][1] * r + g_fr[e][d]) * r + g_coef2[d][i][e][0];
                    // for (size_t p = 0; p < instance.num_MPC_parties; p++) {
                    //     g_fr_shares[e][p][d] = (g_coef_shares[e][d][i][3][p] * r + g_fr_shares[e][p][d]) * r + g_coef_shares[e][d][i][2][p];
                    // }
                    auto r_raw = r.data;
                    __m256i table0 = _mm256_loadu_si256((__m256i *)&(field::GF2E::mul_lut[r_raw][0]));
                    __m256i table1 = _mm256_loadu_si256((__m256i *)&(field::GF2E::mul_lut[r_raw][8]));
                    __m256i table2 = _mm256_loadu_si256((__m256i *)&(field::GF2E::mul_lut[r_raw][16]));
                    __m256i table3 = _mm256_loadu_si256((__m256i *)&(field::GF2E::mul_lut[r_raw][24]));
                    __m256i mask1 = _mm256_set1_epi8(0x0f);
                    __m256i mask2 = _mm256_set1_epi8(0xf0);
                    for (size_t k = 0; k < instance.num_MPC_parties / 16; k++) {
                        __m256i lo, hi, tmp;
                        __m256i t_coef0 = _mm256_loadu_si256((__m256i *)&(g_coef_shares_v[e][d][i][2][k * 32]));
                        __m256i t_coef1 = _mm256_loadu_si256((__m256i *)&(g_coef_shares_v[e][d][i][3][k * 32]));
                        __m256i t_fr    = _mm256_loadu_si256((__m256i *)&(g_fr_shares_v[e][d][k * 32]));

                        // coef1 * r
                        lo = _mm256_and_si256(t_coef1, mask1);  // a0, a2
                        lo = _mm256_shuffle_epi8(table0, lo);   // a0lo, a2hi
                        hi = _mm256_srli_epi64(_mm256_and_si256(t_coef1, mask2), 4);    // a1, a3
                        hi = _mm256_shuffle_epi8(table1, hi);   // a1lo, a3hi
                        tmp = _mm256_xor_si256(hi, lo);
                        t_coef1 = _mm256_permute4x64_epi64(t_coef1, 2 | (3 << 2) | (1 << 6));   // a2,a3; a0,a1
                        lo = _mm256_and_si256(t_coef1, mask1);  // a2, a0
                        lo = _mm256_shuffle_epi8(table2, lo);   // a2lo, a0hi
                        hi = _mm256_srli_epi64(_mm256_and_si256(t_coef1, mask2), 4);    // a3, a1
                        hi = _mm256_shuffle_epi8(table3, hi);   // a3lo, a1hi
                        t_coef1 = _mm256_xor_si256(tmp, _mm256_xor_si256(hi, lo));
                        // + fr
                        t_coef1 = _mm256_xor_si256(t_fr, t_coef1);
                        // * r
                        lo = _mm256_and_si256(t_coef1, mask1);  // a0, a2
                        lo = _mm256_shuffle_epi8(table0, lo);   // a0lo, a2hi
                        hi = _mm256_srli_epi64(_mm256_and_si256(t_coef1, mask2), 4);    // a1, a3
                        hi = _mm256_shuffle_epi8(table1, hi);   // a1lo, a3hi
                        tmp = _mm256_xor_si256(hi, lo);
                        t_coef1 = _mm256_permute4x64_epi64(t_coef1, 2 | (3 << 2) | (1 << 6));   // a2,a3; a0,a1
                        lo = _mm256_and_si256(t_coef1, mask1);  // a2, a0
                        lo = _mm256_shuffle_epi8(table2, lo);   // a2lo, a0hi
                        hi = _mm256_srli_epi64(_mm256_and_si256(t_coef1, mask2), 4);    // a3, a1
                        hi = _mm256_shuffle_epi8(table3, hi);   // a3lo, a1hi
                        t_coef1 = _mm256_xor_si256(tmp, _mm256_xor_si256(hi, lo));
                        // + coef[0]
                        t_coef1 = _mm256_xor_si256(t_coef0, t_coef1);

                        _mm256_storeu_si256((__m256i *)&(g_fr_shares_v[e][d][k * 32]), t_coef1);
                    }

                    int mask = 1 << (bit_len - i - 1);
                    for (int u = 0; u < mask; u++) {
                        int u0 = u << 1;
                        int u1 = u0 + 1;
                        g_AV[e][u] = g_AV[e][u0] + g_AV[e][u1] * r;
                        g_AG[e][u] = g_AG[e][u0] + g_AG[e][u1] * r;
                    }

                    mask = 1 << i;
                    for (int k = 0; k < mask; k++) {
                        g_Sgm[e][k + mask] = g_Sgm[e][k] * r;
                        g_Sgm[e][k] += g_Sgm[e][k + mask];
                    }

                    // update pred
                    field::GF2E one(1);
                    if (i == 0) {
                        g_pred[e][d] *= scalar_d[e][0] * (one + g_rho[d][i][e]) * g_sgm[d][i][e];
                        t_pred[e][d] *= scalar_d[e][1] * (one + g_rho[d][i][e]) * g_sgm[d][i][e];
                    } else {
                        g_pred[e][d] *= g_rho[d][i][e] * g_sgm[d][i][e] * g_rho[d-1][i-1][e] + (one + g_rho[d][i][e]) * (one + g_sgm[d][i][e]) * (one + g_rho[d-1][i-1][e]);
                        t_pred[e][d] *= g_rho[d][i][e] * g_sgm[d][i][e] * g_sgm[d-1][i-1][e] + (one + g_rho[d][i][e]) * (one + g_sgm[d][i][e]) * (one + g_sgm[d-1][i-1][e]);
                    }
                }
            }
            // end of outlayer phase 2
            for (size_t e = 0; e < instance.num_rounds; e++) {
                g_Vs[e][d] = g_AV[e][0];
                g_pred[e][d] += t_pred[e][d];
            }
            // sanity check
            // for (size_t e = 0; e < instance.num_rounds; e++) {
            //     if (g_fr[e][d] != g_Vr[e][d] * g_Vs[e][d] * g_pred[e][d]) {
            //         throw std::runtime_error("sanity check group gkr layer failed.");
            //     }
            // }
        }

#ifdef TIMING
    tmp_time = timing_read(&ctx);
    printf("GGKR other layer time: %ld\n", tmp_time - start_time);
    start_time = timing_read(&ctx);
#endif

        // compute shares of Vr Vs on input layer
        for (size_t e = 0; e < instance.num_rounds; e++) {
            // for (int d = 0; d < depth - 1; d++)
            //     for (size_t k = 0; k < instance.num_MPC_parties / 16; k++)
            //         for (size_t p = 0; p < 16; p++) {
            //             g_Vr_shares[e][k * 16 + p][d] = field::GF2E(
            //                     g_Vr_shares_v[e][d][k * 32 + p] |
            //                     ((uint16_t)g_Vr_shares_v[e][d][k * 32 + p + 16] << 8));
            //             g_Vs_shares[e][k * 16 + p][d] = field::GF2E(
            //                     g_Vs_shares_v[e][d][k * 32 + p] |
            //                     ((uint16_t)g_Vs_shares_v[e][d][k * 32 + p + 16] << 8));
            //             g_fr_shares[e][k * 16 + p][d] = field::GF2E(
            //                     g_fr_shares_v[e][d][k * 32 + p] |
            //                     ((uint16_t)g_fr_shares_v[e][d][k * 32 + p + 16] << 8));
            //        }
            // for (size_t k = 0; k < instance.num_MPC_parties / 16; k++)
            //     for (size_t p = 0; p < 16; p++)
            //         g_fr_shares[e][k * 16 + p][depth-1] = field::GF2E(
            //                 g_fr_shares_v[e][depth-1][k * 32 + p] |
            //                 ((uint16_t)g_fr_shares_v[e][depth-1][k * 32 + p + 16] << 8));

            std::vector<std::vector<uint8_t>> tmpr(gkey.size());
            std::vector<std::vector<uint8_t>> tmps(gkey.size());
            for (size_t g = 0; g < gkey.size(); g++) {
                tmpr[g].resize(instance.num_MPC_parties * 2);
                tmps[g].resize(instance.num_MPC_parties * 2);
                for (size_t k = 0; k < instance.num_MPC_parties / 16; k++) {
                    for (size_t p = 0; p < 16; p++) {
                        auto tmp = field::lift(keytest_share_v[e][g][k * 16 + p]);
                        tmpr[g][k * 32 + p] = tmp & 255;
                        tmps[g][k * 32 + p] = tmp & 255;
                        tmpr[g][k * 32 + p + 16] = tmp >> 8;
                        tmps[g][k * 32 + p + 16] = tmp >> 8;
                    }
                }
            }

            for (int i = 0; i < depth; i++) {
                int mask = 1 << (depth - i - 1);
                int r_raw = g_rho[d-1][i][e].data;
                int s_raw = g_sgm[d-1][i][e].data;
                const __m256i tabler0 = _mm256_loadu_si256((__m256i *)&(field::GF2E::mul_lut[r_raw][0]));
                const __m256i tabler1 = _mm256_loadu_si256((__m256i *)&(field::GF2E::mul_lut[r_raw][8]));
                const __m256i tabler2 = _mm256_loadu_si256((__m256i *)&(field::GF2E::mul_lut[r_raw][16]));
                const __m256i tabler3 = _mm256_loadu_si256((__m256i *)&(field::GF2E::mul_lut[r_raw][24]));

                const __m256i tables0 = _mm256_loadu_si256((__m256i *)&(field::GF2E::mul_lut[s_raw][0]));
                const __m256i tables1 = _mm256_loadu_si256((__m256i *)&(field::GF2E::mul_lut[s_raw][8]));
                const __m256i tables2 = _mm256_loadu_si256((__m256i *)&(field::GF2E::mul_lut[s_raw][16]));
                const __m256i tables3 = _mm256_loadu_si256((__m256i *)&(field::GF2E::mul_lut[s_raw][24]));

                const __m256i mask1 = _mm256_set1_epi8(0x0f);
                const __m256i mask2 = _mm256_set1_epi8(0xf0);
                for (int u = 0; u < mask; u++) {
                    int u0 = u << 1;
                    int u1 = u0 + 1;
                    for (size_t k = 0; k < instance.num_MPC_parties / 16; k++) {
                        __m256i lo, hi, tmp, t_u0, t_u1;

                        t_u0 = _mm256_loadu_si256((__m256i *)&(tmpr[u0][k * 32]));
                        t_u1 = _mm256_loadu_si256((__m256i *)&(tmpr[u1][k * 32]));
                        t_u1 = _mm256_xor_si256(t_u0, t_u1);
                        // (t_u1 + t_u0) * r
                        lo = _mm256_and_si256(t_u1, mask1);  // a0, a2
                        lo = _mm256_shuffle_epi8(tabler0, lo);   // a0lo, a2hi
                        hi = _mm256_srli_epi64(_mm256_and_si256(t_u1, mask2), 4);    // a1, a3
                        hi = _mm256_shuffle_epi8(tabler1, hi);   // a1lo, a3hi
                        tmp = _mm256_xor_si256(hi, lo);
                        t_u1 = _mm256_permute4x64_epi64(t_u1, 2 | (3 << 2) | (1 << 6));   // a2,a3; a0,a1
                        lo = _mm256_and_si256(t_u1, mask1);  // a2, a0
                        lo = _mm256_shuffle_epi8(tabler2, lo);   // a2lo, a0hi
                        hi = _mm256_srli_epi64(_mm256_and_si256(t_u1, mask2), 4);    // a3, a1
                        hi = _mm256_shuffle_epi8(tabler3, hi);   // a3lo, a1hi
                        t_u1 = _mm256_xor_si256(tmp, _mm256_xor_si256(hi, lo));
                        // + t_u0
                        t_u1 = _mm256_xor_si256(t_u0, t_u1);
                        _mm256_storeu_si256((__m256i *)&(tmpr[u][k * 32]), t_u1);

                        t_u0 = _mm256_loadu_si256((__m256i *)&(tmps[u0][k * 32]));
                        t_u1 = _mm256_loadu_si256((__m256i *)&(tmps[u1][k * 32]));
                        t_u1 = _mm256_xor_si256(t_u0, t_u1);
                        // (t_u1 + t_u0) * s
                        lo = _mm256_and_si256(t_u1, mask1);  // a0, a2
                        lo = _mm256_shuffle_epi8(tables0, lo);   // a0lo, a2hi
                        hi = _mm256_srli_epi64(_mm256_and_si256(t_u1, mask2), 4);    // a1, a3
                        hi = _mm256_shuffle_epi8(tables1, hi);   // a1lo, a3hi
                        tmp = _mm256_xor_si256(hi, lo);
                        t_u1 = _mm256_permute4x64_epi64(t_u1, 2 | (3 << 2) | (1 << 6));   // a2,a3; a0,a1
                        lo = _mm256_and_si256(t_u1, mask1);  // a2, a0
                        lo = _mm256_shuffle_epi8(tables2, lo);   // a2lo, a0hi
                        hi = _mm256_srli_epi64(_mm256_and_si256(t_u1, mask2), 4);    // a3, a1
                        hi = _mm256_shuffle_epi8(tables3, hi);   // a3lo, a1hi
                        t_u1 = _mm256_xor_si256(tmp, _mm256_xor_si256(hi, lo));
                        // + t_u0
                        t_u1 = _mm256_xor_si256(t_u0, t_u1);
                        _mm256_storeu_si256((__m256i *)&(tmps[u][k * 32]), t_u1);
                    }
                }
            }
            for (size_t p = 0; p < instance.num_MPC_parties * 2; p++) {
                g_Vr_shares_v[e][depth-1][p] = tmpr[0][p];
                g_Vs_shares_v[e][depth-1][p] = tmps[0][p];
            }
            // sanity_check
            // for (int d = 0; d < depth; d++) {
            //     field::GF2E tmp(0);
            //     for (size_t p = 0; p < instance.num_MPC_parties; p++)
            //         tmp += g_Vr_shares[e][p][d];
            //     printf("check Vr shares %d\n", tmp == g_Vr[e][d]);
            //     tmp = field::GF2E(0);
            //     for (size_t p = 0; p < instance.num_MPC_parties; p++)
            //         tmp += g_Vs_shares[e][p][d];
            //     printf("check Vs shares %d\n", tmp == g_Vs[e][d]);
            // }
        }

        // prove that sum(g_Vr_shares) * sum(g_Vs_shares) * g_pred == sum(g_fr_share)
        // merge g_pred into g_Vs_shares
        for (size_t e = 0; e < instance.num_rounds; e++)
            for (int d = 0; d < depth; d++) {
                g_Vs[e][d] *= g_pred[e][d];
                int r_raw = g_pred[e][d].data;
                const __m256i table0 = _mm256_loadu_si256((__m256i *)&(field::GF2E::mul_lut[r_raw][0]));
                const __m256i table1 = _mm256_loadu_si256((__m256i *)&(field::GF2E::mul_lut[r_raw][8]));
                const __m256i table2 = _mm256_loadu_si256((__m256i *)&(field::GF2E::mul_lut[r_raw][16]));
                const __m256i table3 = _mm256_loadu_si256((__m256i *)&(field::GF2E::mul_lut[r_raw][24]));
                const __m256i mask1 = _mm256_set1_epi8(0x0f);
                const __m256i mask2 = _mm256_set1_epi8(0xf0);
                for (size_t k = 0; k < instance.num_MPC_parties / 16; k++) {
                    __m256i lo, hi, tmp;
                    __m256i t_coef2 = _mm256_loadu_si256((__m256i *)&(g_Vs_shares_v[e][d][k * 32]));
                    // coef[2] * r
                    lo = _mm256_and_si256(t_coef2, mask1);  // a0, a2
                    lo = _mm256_shuffle_epi8(table0, lo);   // a0lo, a2hi
                    hi = _mm256_srli_epi64(_mm256_and_si256(t_coef2, mask2), 4);    // a1, a3
                    hi = _mm256_shuffle_epi8(table1, hi);   // a1lo, a3hi
                    tmp = _mm256_xor_si256(hi, lo);
                    t_coef2 = _mm256_permute4x64_epi64(t_coef2, 2 | (3 << 2) | (1 << 6));   // a2,a3; a0,a1
                    lo = _mm256_and_si256(t_coef2, mask1);  // a2, a0
                    lo = _mm256_shuffle_epi8(table2, lo);   // a2lo, a0hi
                    hi = _mm256_srli_epi64(_mm256_and_si256(t_coef2, mask2), 4);    // a3, a1
                    hi = _mm256_shuffle_epi8(table3, hi);   // a3lo, a1hi
                    t_coef2 = _mm256_xor_si256(tmp, _mm256_xor_si256(hi, lo));

                    _mm256_storeu_si256((__m256i *)&(g_Vs_shares_v[e][d][k * 32]), t_coef2);
                }
            }

#ifdef TIMING
    tmp_time = timing_read(&ctx);
    printf("GGKR Vr Vs fr time: %ld\n", tmp_time - start_time);
    start_time = timing_read(&ctx);
#endif

        for (size_t e = 0; e < instance.num_rounds; e++) {
            g_fr_poly_share[e].resize(instance.num_MPC_parties);
            g_Vr_poly_share[e].resize(instance.num_MPC_parties);
            g_Vs_poly_share[e].resize(instance.num_MPC_parties);
            g_P_share[e].resize(instance.num_MPC_parties);
            g_P[e].resize(2 * depth + 1);
            g_P_deltas[e].resize(depth + 1);
            for (size_t p = 0; p < instance.num_MPC_parties; p++) {
                auto random_share = random_tapes.get_bytes(e, p,
                                    random_offset,
                                    3 * instance.lambda +
                                    (depth + 1) * instance.lambda);
                int idx = (p / 16) * 32 + (p % 16);
                g_fr_shares_v[e][depth][idx] = *(uint8_t *)(random_share.data());
                g_Vr_shares_v[e][depth][idx] = *(uint8_t *)(random_share.data() + 2);
                g_Vs_shares_v[e][depth][idx] = *(uint8_t *)(random_share.data() + 4);
                g_fr_shares_v[e][depth][idx + 16] = *(uint8_t *)(random_share.data() + 1);
                g_Vr_shares_v[e][depth][idx + 16] = *(uint8_t *)(random_share.data() + 3);
                g_Vs_shares_v[e][depth][idx + 16] = *(uint8_t *)(random_share.data() + 5);
                // WARN: not work for GF2E except GF2^16
                g_fr[e][depth] += field::GF2E(*(uint16_t *)(random_share.data()));
                g_Vr[e][depth] += field::GF2E(*(uint16_t *)(random_share.data() + 2));
                g_Vs[e][depth] += field::GF2E(*(uint16_t *)(random_share.data() + 4));
                g_fr_poly_share[e][p] = interpolate_with_precomputation(precomputation_for_zero_to_d, g_fr_shares_v[e], idx);
                g_Vr_poly_share[e][p] = interpolate_with_precomputation(precomputation_for_zero_to_d, g_Vr_shares_v[e], idx);
                g_Vs_poly_share[e][p] = interpolate_with_precomputation(precomputation_for_zero_to_d, g_Vs_shares_v[e], idx);

                g_P_share[e][p].resize(2 * depth + 1);
                for (int i = depth; i < 2 * depth + 1; i++) {
                    g_P_share[e][p][i].from_bytes(random_share.data() + (3 + i - depth) * instance.lambda);
                    g_P_deltas[e][i - depth] += g_P_share[e][p][i];
                }
            }
            g_fr_poly[e] = interpolate_with_precomputation(precomputation_for_zero_to_d, g_fr[e]);
            g_Vr_poly[e] = interpolate_with_precomputation(precomputation_for_zero_to_d, g_Vr[e]);
            g_Vs_poly[e] = interpolate_with_precomputation(precomputation_for_zero_to_d, g_Vs[e]);
            auto g_P = g_fr_poly[e] + g_Vr_poly[e] * g_Vs_poly[e];
            for (int i = depth; i < 2 * depth + 1; i++) {
                // calculate offset
                field::GF2E i_element = x_values_for_interpolation_zero_to_2d[i];
                g_P_deltas[e][i - depth] = eval(g_P, i_element)
                    - g_P_deltas[e][i - depth];
                // adjust first share
                g_P_share[e][0][i] += g_P_deltas[e][i - depth];
            }
        }
        // tape size sanity check
        // random_offset += (4 + depth) * instance.lambda;
        // printf("check tape size %ld, %d, %d\n", random_tape_size, random_offset, random_tape_size == random_offset);
    }

#ifdef TIMING
    tmp_time = timing_read(&ctx);
    printf("GGKR poly compute time: %ld\n", tmp_time - start_time);
    start_time = timing_read(&ctx);
#endif

    // challenge the checking polynomials
    std::vector<uint8_t> h_4 = phase_2_commitment(instance, salt, h_i, g_P_deltas);
    // expand challenge hash to depth values
    std::vector<field::GF2E> g_forbidden_challenge_values = field::get_first_n_field_elements(depth);
    std::vector<field::GF2E> g_R_es = phase_2_expand(instance, h_4, g_forbidden_challenge_values);

    // commit to the views of the checking protocol
    std::vector<field::GF2E> g_a(instance.num_rounds);
    std::vector<field::GF2E> g_b(instance.num_rounds);
    std::vector<field::GF2E> g_c(instance.num_rounds);
    std::vector<std::vector<field::GF2E>> g_a_shares(instance.num_rounds);
    std::vector<std::vector<field::GF2E>> g_b_shares(instance.num_rounds);
    std::vector<std::vector<field::GF2E>> g_c_shares(instance.num_rounds);
    std::vector<std::vector<field::GF2E>> g_d_shares(instance.num_rounds);
    std::vector<field::GF2E> lagrange_polys_evaluated_at_Re_d(depth + 1);
    std::vector<field::GF2E> lagrange_polys_evaluated_at_Re_2d(2 * depth + 1);

    for (size_t e = 0; e < instance.num_rounds; e++) {
        g_a_shares[e].resize(instance.num_MPC_parties);
        g_b_shares[e].resize(instance.num_MPC_parties);
        g_c_shares[e].resize(instance.num_MPC_parties);
        g_d_shares[e].resize(instance.num_MPC_parties);
        for (int k = 0; k < depth + 1; k++)
            lagrange_polys_evaluated_at_Re_d[k] = eval(precomputation_for_zero_to_d[k], g_R_es[e]);
        for (int k = 0; k < 2 * depth + 1; k++)
            lagrange_polys_evaluated_at_Re_2d[k] = eval(precomputation_for_zero_to_2d[k], g_R_es[e]);

        for (size_t p = 0; p < instance.num_MPC_parties; p++) {
            int idx = (p / 16) * 32 + (p % 16);
            g_a_shares[e][p] = dot_product(lagrange_polys_evaluated_at_Re_d, g_fr_shares_v[e], idx);
            g_b_shares[e][p] = dot_product(lagrange_polys_evaluated_at_Re_d, g_Vr_shares_v[e], idx);
            g_c_shares[e][p] = dot_product(lagrange_polys_evaluated_at_Re_d, g_Vs_shares_v[e], idx);
            g_d_shares[e][p] = dot_product(lagrange_polys_evaluated_at_Re_2d, g_P_share[e][p]);
        }

        // open d_e and a,b,c values
        for (size_t p = 0; p < instance.num_MPC_parties; p++) {
            g_a[e] += g_a_shares[e][p];
            g_b[e] += g_b_shares[e][p];
            g_c[e] += g_c_shares[e][p];
        }
    }
    // sanity check
    for (size_t e = 0; e < instance.num_rounds; e++) {
        field::GF2E g_d(0);
        for (size_t p = 0; p < instance.num_MPC_parties; p++) {
            g_d += g_d_shares[e][p];
        }
    }

#ifdef TIMING
    tmp_time = timing_read(&ctx);
    printf("GGKR poly commit time: %ld\n", tmp_time - start_time);
    start_time = timing_read(&ctx);
#endif

    /////////////////////////////////////////////////////////////////////////////
    // phase 6: challenge the views of the checking protocol
    /////////////////////////////////////////////////////////////////////////////

    std::vector<uint8_t> h_5 = phase_3_commitment(
            instance, salt, h_4, g_d_shares, g_a, g_a_shares, g_b, g_b_shares, g_c, g_c_shares);

    std::vector<uint16_t> missing_parties = phase_3_expand(instance, h_5);

    /////////////////////////////////////////////////////////////////////////////
    // phase 7: Open the views of the checking protocol
    /////////////////////////////////////////////////////////////////////////////
    std::vector<reveal_list_t> seeds;
    for (size_t e = 0; e < instance.num_rounds; e++) {
        seeds.push_back(
                seed_trees[e].reveal_all_but(missing_parties[e]));
    }
    // build signature
    std::vector<dubhe_repetition_group_proof_t> proofs;
    bit_len = instance.aes_params.bit_len;
    for (size_t e = 0; e < instance.num_rounds; e++) {
        size_t missing_party = missing_parties[e];
        std::vector<uint8_t> commitment(instance.digest_size);
        auto missing_commitment =
            party_seed_commitments.get(e, missing_party);
        std::copy(std::begin(missing_commitment), std::end(missing_commitment),
                std::begin(commitment));
        std::vector<std::vector<field::GF2E>> coef1(bit_len);
        std::vector<std::vector<field::GF2E>> coef2(bit_len);
        std::vector<std::vector<std::vector<field::GF2E>>> g_coef1(depth);
        std::vector<std::vector<std::vector<field::GF2E>>> g_coef2(depth);
        std::vector<field::GF2E> g_Vr(depth - 1);
        std::vector<field::GF2E> g_Vs(depth - 1);
        for (int i = 0; i < bit_len; i++) {
            coef1[i] = coef1_deltas[i][e];
            coef2[i] = coef2_deltas[i][e];
        }
        for (int d = 0; d < depth; d++) {
            g_coef1[d].resize(d + 1);
            g_coef2[d].resize(d + 1);
            for (int i = 0; i < d + 1; i++) {
                g_coef1[d][i] = g_coef1_deltas[d][i][e];    // vector of 2 coefs
                g_coef2[d][i] = g_coef2_deltas[d][i][e];    // vector of 2 coefs
            }
            if (d != depth - 1) {
                g_Vr[d] = g_Vr_deltas[d][e];
                g_Vs[d] = g_Vs_deltas[d][e];
            }
        }
        dubhe_repetition_group_proof_t proof{
            seeds[e],
            commitment,
            rep_key_deltas[e],
            rep_pt_deltas[e],
            rep_t_deltas[e],
            coef1,
            coef2,
            P_deltas[e],
            a[e],
            b[e],
            c[e],
            g_coef1,
            g_coef2,
            g_Vr,
            g_Vs,
            g_P_deltas[e],
            g_a[e],
            g_b[e],
            g_c[e],
        };

        // sanity check c = sum_j a*b
        // if (a[e] + pred[e] * b[e] * (field::GF2E(1) - b[e] * c[e]) != d[e])
        //     printf("pcp check failed\n");

        // field::GF2E accum_p;
        // field::GF2E accum_q;
        // for (size_t j = 0; j < instance.m1; j++) {
        //   accum_p += a[e][j] * (r_ejs[repetition][j] +  a[repetition][j] * b[repetition][j]);
        //   accum_q += b[repetition][j] * (r_ejs[repetition][j] +  a[repetition][j] * b[repetition][j]);
        // }
        // if (accum_p != c[repetition] || accum_q != d[repetition])
        //   throw std::runtime_error("final sanity check is wrong");
        proofs.push_back(proof);
    }

    dubhe_group_signature_t signature{salt, h_1, h_5, proofs};

#ifdef TIMING
    tmp_time = timing_read(&ctx);
    printf("reveal seeds and construct signature time: %ld\n", tmp_time - start_time);
    start_time = timing_read(&ctx);
#endif

    return signature;
}


bool dubhe_group_verify(const dubhe_instance_t &instance,
                    const dubhe_group_signature_t &signature,
                    const std::vector<dubhe_keypair_t> &gkey,
                    const uint8_t *message, size_t message_len) {
  // init modulus of extension field F_{2^{8\lambda}}
  // F::init_extension_field();
    field::GF2E::init_extension_field(instance);


  // std::vector<uint8_t> pt(instance.aes_params.block_size *
  //                         instance.aes_params.num_blocks),
  //   ct(instance.aes_params.block_size * instance.aes_params.num_blocks);
  // memcpy(pt.data(), pk.data(), pt.size());
  // memcpy(ct.data(), pk.data(), ct.size());

  // do parallel repetitions
  // create seed trees and random tapes
  std::vector<SeedTree> seed_trees;
  seed_trees.reserve(instance.num_rounds);

  int bit_len = instance.aes_params.bit_len;
  int depth = ceil_log2(gkey.size());
  size_t random_tape_size =
      instance.aes_params.key_size +
      instance.aes_params.block_size * instance.aes_params.num_blocks +
      instance.aes_params.num_sboxes +
      3 * bit_len * instance.lambda +
      2 * bit_len * instance.lambda +
      3 * instance.lambda + 3 * instance.lambda +
      2 * (depth - 1) * instance.lambda +
      2 * depth * (depth + 1) * instance.lambda +
      (4 + depth) * instance.lambda;
  RandomTapes random_tapes(instance.num_rounds, instance.num_MPC_parties,
                           random_tape_size);
  RepByteContainer party_seed_commitments(
      instance.num_rounds, instance.num_MPC_parties, instance.digest_size);

  // sumcheck
  std::vector<uint8_t> h_i = signature.h_1;

    std::vector<std::vector<std::vector<field::GF2E>>> coef1_deltas(bit_len);
    std::vector<std::vector<std::vector<field::GF2E>>> coef2_deltas(bit_len);
    std::vector<std::vector<field::GF2E>> rho;
    std::vector<std::vector<field::GF2E>> sgm;
    std::vector<std::vector<field::GF2E>> tau = phase_sumcheck0_expand(instance, h_i);
    std::vector<field::GF2E> pred(instance.num_rounds, field::GF2E(1));
    for (int i = 0; i < bit_len; i++) {
        coef1_deltas[i].resize(instance.num_rounds);
        for (size_t e = 0; e < instance.num_rounds; e++) {
            coef1_deltas[i][e] = signature.proofs[e].coef1_delta[i];
        }
        h_i = phase_sumcheck_commitment(instance, signature.salt, h_i, coef1_deltas[i]);
        std::vector<field::GF2E> rho_i = phase_sumcheck_expand(instance, h_i);
        rho.push_back(rho_i);
    }

    for (int i = 0; i < bit_len; i++) {
        coef2_deltas[i].resize(instance.num_rounds);
        for (size_t e = 0; e < instance.num_rounds; e++)
            coef2_deltas[i][e] = signature.proofs[e].coef2_delta[i];
        h_i = phase_sumcheck_commitment(instance, signature.salt, h_i, coef2_deltas[i]);
        std::vector<field::GF2E> sgm_i = phase_sumcheck_expand(instance, h_i);
        sgm.push_back(sgm_i);
    }
    for (size_t e = 0; e < instance.num_rounds; e++) {
        for (int i = 0; i < bit_len - 1; i++) {
            pred[e] *= tau[i][e] * rho[i][e] + (tau[i][e] + rho[i][e] + field::GF2E(1)) * (sgm[i][e] + field::GF2E(1));
        }
        pred[e] *= tau[bit_len-1][e] * rho[bit_len-1][e] + (tau[bit_len-1][e] + rho[bit_len-1][e] + field::GF2E(1)) * sgm[bit_len-1][e];
    }

  // recompute h_2
  std::vector<std::vector<field::GF2E>> P_deltas;
  for (const dubhe_repetition_group_proof_t &proof : signature.proofs) {
    P_deltas.push_back(proof.P_delta);
  }
  std::vector<uint8_t> h_2 =
      phase_2_commitment(instance, signature.salt, h_i, P_deltas);


  // compute challenges based on hashes

  // h2 expansion
  std::vector<field::GF2E> forbidden_challenge_values = field::get_first_n_field_elements(1);
  std::vector<field::GF2E> R_es = phase_2_expand(instance, h_2, forbidden_challenge_values);
  // h3 expansion already happened in deserialize to get missing parties
  std::vector<uint16_t> missing_parties = phase_3_expand(instance, signature.h_5);

  // rebuild SeedTrees
  for (size_t repetition = 0; repetition < instance.num_rounds; repetition++) {
    const dubhe_repetition_group_proof_t &proof = signature.proofs[repetition];
    // regenerate generate seed tree for the N parties (except the missing
    // one)
    if (missing_parties[repetition] != proof.reveallist.second)
      throw std::runtime_error(
          "modified signature between deserialization and verify");
    seed_trees.push_back(SeedTree(proof.reveallist, instance.num_MPC_parties,
                                  signature.salt, repetition));
    // commit to each party's seed, fill up missing one with data from proof
    {
      std::vector<uint8_t> dummy(instance.seed_size);
      size_t party = 0;
      for (; party < (instance.num_MPC_parties / 4) * 4; party += 4) {
        auto seed0 = seed_trees[repetition].get_leaf(party).value_or(dummy);
        auto seed1 = seed_trees[repetition].get_leaf(party + 1).value_or(dummy);
        auto seed2 = seed_trees[repetition].get_leaf(party + 2).value_or(dummy);
        auto seed3 = seed_trees[repetition].get_leaf(party + 3).value_or(dummy);
        commit_to_4_party_seeds(
            instance, seed0, seed1, seed2, seed3, signature.salt, repetition,
            party, party_seed_commitments.get(repetition, party),
            party_seed_commitments.get(repetition, party + 1),
            party_seed_commitments.get(repetition, party + 2),
            party_seed_commitments.get(repetition, party + 3));
      }
      for (; party < instance.num_MPC_parties; party++) {
        if (party != missing_parties[repetition]) {
          commit_to_party_seed(instance,
                               seed_trees[repetition].get_leaf(party).value(),
                               signature.salt, repetition, party,
                               party_seed_commitments.get(repetition, party));
        }
      }
    }
    auto com =
        party_seed_commitments.get(repetition, missing_parties[repetition]);
    std::copy(std::begin(proof.C_e), std::end(proof.C_e), std::begin(com));

    // create random tape for each party, dummy one for missing party
    {
      size_t party = 0;
      std::vector<uint8_t> dummy(instance.seed_size);
      for (; party < (instance.num_MPC_parties / 4) * 4; party += 4) {
        random_tapes.generate_4_tapes(
            repetition, party, signature.salt,
            seed_trees[repetition].get_leaf(party).value_or(dummy),
            seed_trees[repetition].get_leaf(party + 1).value_or(dummy),
            seed_trees[repetition].get_leaf(party + 2).value_or(dummy),
            seed_trees[repetition].get_leaf(party + 3).value_or(dummy));
      }
      for (; party < instance.num_MPC_parties; party++) {
        random_tapes.generate_tape(
            repetition, party, signature.salt,
            seed_trees[repetition].get_leaf(party).value_or(dummy));
      }
    }
  }

  /////////////////////////////////////////////////////////////////////////////
  // recompute commitments to executions of AES
  /////////////////////////////////////////////////////////////////////////////

  RepByteContainer rep_shared_keys(instance.num_rounds,
                                   instance.num_MPC_parties,
                                   instance.aes_params.key_size);
  RepByteContainer rep_shared_pt(instance.num_rounds,
                                 instance.num_MPC_parties,
                                 instance.aes_params.block_size * instance.aes_params.num_blocks);
  RepByteContainer rep_shared_ct(instance.num_rounds,
                                 instance.num_MPC_parties,
                                 instance.aes_params.block_size * instance.aes_params.num_blocks);
  RepByteContainer rep_shared_s(instance.num_rounds, instance.num_MPC_parties,
                                instance.aes_params.num_sboxes);
  RepByteContainer rep_shared_t(instance.num_rounds, instance.num_MPC_parties,
                                instance.aes_params.num_sboxes);

  for (size_t repetition = 0; repetition < instance.num_rounds; repetition++) {
    const dubhe_repetition_group_proof_t &proof = signature.proofs[repetition];

    // generate sharing of secret key
    for (size_t party = 0; party < instance.num_MPC_parties; party++) {
      auto shared_key = rep_shared_keys.get(repetition, party);
      auto random_key_share =
          random_tapes.get_bytes(repetition, party, 0, shared_key.size());
      std::copy(std::begin(random_key_share), std::end(random_key_share),
                std::begin(shared_key));
    }

    // fix first share
    auto first_key_share = rep_shared_keys.get(repetition, 0);
    std::transform(std::begin(proof.sk_delta), std::end(proof.sk_delta),
                   std::begin(first_key_share), std::begin(first_key_share),
                   std::bit_xor<uint8_t>());

    // generate sharing of public key
    for (size_t party = 0; party < instance.num_MPC_parties; party++) {
      auto shared_pt = rep_shared_pt.get(repetition, party);
      auto random_pt_share =
          random_tapes.get_bytes(repetition, party, instance.aes_params.key_size, shared_pt.size());
      std::copy(std::begin(random_pt_share), std::end(random_pt_share),
                std::begin(shared_pt));
    }

    // fix first share
    auto first_pt_share = rep_shared_pt.get(repetition, 0);
    std::transform(std::begin(proof.pk_delta), std::end(proof.pk_delta),
                   std::begin(first_pt_share), std::begin(first_pt_share),
                   std::bit_xor<uint8_t>());

    // generate sharing of t values
    for (size_t party = 0; party < instance.num_MPC_parties; party++) {
      auto shared_t = rep_shared_t.get(repetition, party);
      auto random_t_shares = random_tapes.get_bytes(
          repetition, party, instance.aes_params.key_size + instance.aes_params.block_size * instance.aes_params.num_blocks,
          instance.aes_params.num_sboxes);
      std::copy(std::begin(random_t_shares), std::end(random_t_shares),
                std::begin(shared_t));
    }
    // fix first share
    auto first_shared_t = rep_shared_t.get(repetition, 0);
    std::transform(std::begin(proof.t_delta), std::end(proof.t_delta),
                   std::begin(first_shared_t), std::begin(first_shared_t),
                   std::bit_xor<uint8_t>());

    // get shares of sbox inputs by executing MPC AES
    auto ct_shares = rep_shared_ct.get_repetition(repetition);
    auto shared_s = rep_shared_s.get_repetition(repetition);
    auto pt_shares = rep_shared_pt.get_repetition(repetition);

    if (instance.aes_params.key_size == 16)
      AES128::group_aes_128_s_shares(
              rep_shared_keys.get_repetition(repetition),
              rep_shared_t.get_repetition(repetition), pt_shares,
              ct_shares, shared_s);
    else if (instance.aes_params.key_size == 24)
      AES192::group_aes_192_s_shares(
              rep_shared_keys.get_repetition(repetition),
              rep_shared_t.get_repetition(repetition), pt_shares,
              ct_shares, shared_s);
    else if (instance.aes_params.key_size == 32)
      AES256::group_aes_256_s_shares(
              rep_shared_keys.get_repetition(repetition),
              rep_shared_t.get_repetition(repetition), pt_shares,
              ct_shares, shared_s);
    else
      throw std::runtime_error("invalid parameters");

    // calculate missing output broadcast
    // std::copy(ct.begin(), ct.end(),
    //           ct_shares[missing_parties[repetition]].begin());
    // for (size_t party = 0; party < instance.num_MPC_parties; party++) {
    //   if (party != missing_parties[repetition])
    //     std::transform(std::begin(ct_shares[party]), std::end(ct_shares[party]),
    //                    std::begin(ct_shares[missing_parties[repetition]]),
    //                    std::begin(ct_shares[missing_parties[repetition]]),
    //                    std::bit_xor<uint8_t>());
    // }
  }

  /////////////////////////////////////////////////////////////////////////////
  // recompute shares of polynomials
  /////////////////////////////////////////////////////////////////////////////

    // std::vector<std::vector<std::vector<field::GF2E>>> inputs_rho_share(instance.num_rounds);
    // std::vector<std::vector<std::vector<field::GF2E>>> inputs_sgm_share(instance.num_rounds);
    // RepContainer<std::array<field::GF2E, 3>> coef1_shares(instance.num_rounds, instance.num_MPC_parties, bit_len);
    // RepContainer<std::array<field::GF2E, 2>> coef2_shares(instance.num_rounds, instance.num_MPC_parties, bit_len);
    RepContainer<field::GF2E> Vr_share(instance.num_rounds, instance.num_MPC_parties, 2);
    RepContainer<field::GF2E> Vs_share(instance.num_rounds, instance.num_MPC_parties, 2);
    RepContainer<field::GF2E> fr_share(instance.num_rounds, instance.num_MPC_parties, 2);

    std::vector<std::vector<std::vector<uint8_t>>> inputs_rho_share_v(instance.num_rounds);
    std::vector<std::vector<std::vector<uint8_t>>> inputs_sgm_share_v(instance.num_rounds);
    std::vector<std::vector<std::array<std::vector<uint8_t>, 5>>> coef_shares_v(instance.num_rounds);
    std::vector<std::vector<uint8_t>> fr_shares_v(instance.num_rounds);

    for (size_t e = 0; e < instance.num_rounds; e++) {
        coef_shares_v[e].resize(bit_len);
        fr_shares_v[e].resize(instance.num_MPC_parties * 2);
        inputs_rho_share_v[e].resize(1 << bit_len);
        inputs_sgm_share_v[e].resize(1 << bit_len);
        for (int i = 0; i < bit_len; i++)
            for (int j = 0; j < 5; j++)
                coef_shares_v[e][i][j].resize(instance.num_MPC_parties * 2);
        for (int i = 0; i < (1 << bit_len); i++) {
            inputs_rho_share_v[e][i].resize(instance.num_MPC_parties * 2);
            inputs_sgm_share_v[e][i].resize(instance.num_MPC_parties * 2);
        }
        for (size_t i = 0; i < instance.aes_params.num_sboxes; i++) {
            size_t j = (1 << (bit_len - 1)) + i;
            for (size_t k = 0; k < instance.num_MPC_parties / 16; k++) {
                for (size_t p = 0; p < 16; p++) {
                    auto si = field::lift_uint8_t(rep_shared_s.get(e, k * 16 + p)[i]).data;
                    auto ti = field::lift_uint8_t(rep_shared_t.get(e, k * 16 + p)[i]).data;
                    inputs_rho_share_v[e][i][k * 32 + p] = si & 255;
                    inputs_rho_share_v[e][i][k * 32 + 16 + p] = si >> 8;
                    inputs_rho_share_v[e][j][k * 32 + p] = ti & 255;
                    inputs_rho_share_v[e][j][k * 32 + 16 + p] = ti >> 8;
                    inputs_sgm_share_v[e][i][k * 32 + 16 + p] = si >> 8;
                    inputs_sgm_share_v[e][i][k * 32 + p] = si & 255;
                    inputs_sgm_share_v[e][j][k * 32 + 16 + p] = ti >> 8;
                    inputs_sgm_share_v[e][j][k * 32 + p] = ti & 255;
                }
            }
        }
    }

    // for (size_t e = 0; e < instance.num_rounds; e++) {
    //     inputs_rho_share[e].resize(instance.num_MPC_parties);
    //     inputs_sgm_share[e].resize(instance.num_MPC_parties);
    //     for (size_t p = 0; p < instance.num_MPC_parties; p++) {
    //         inputs_rho_share[e][p].resize(1 << bit_len);
    //         inputs_sgm_share[e][p].resize(1 << bit_len);
    //         auto shared_s = rep_shared_s.get(e, p);
    //         auto shared_t = rep_shared_t.get(e, p);
    //         for (size_t i = 0; i < instance.aes_params.num_sboxes; i++) {
    //             size_t j = (1 << (bit_len - 1)) + i;
    //             inputs_rho_share[e][p][i] = field::lift_uint8_t(shared_s[i]);
    //             inputs_rho_share[e][p][j] = field::lift_uint8_t(shared_t[i]);
    //             inputs_sgm_share[e][p][i] = field::lift_uint8_t(shared_s[i]);
    //             inputs_sgm_share[e][p][j] = field::lift_uint8_t(shared_t[i]);
    //         }
    //     }
    // }
    for (int i = 0; i < bit_len; i++)
        for (size_t e = 0; e < instance.num_rounds; e++) {
            auto r = rho[i][e];
            auto r_raw = r.data;
            const __m256i table0 = _mm256_loadu_si256((__m256i *)&(field::GF2E::mul_lut[r_raw][0]));
            const __m256i table1 = _mm256_loadu_si256((__m256i *)&(field::GF2E::mul_lut[r_raw][8]));
            const __m256i table2 = _mm256_loadu_si256((__m256i *)&(field::GF2E::mul_lut[r_raw][16]));
            const __m256i table3 = _mm256_loadu_si256((__m256i *)&(field::GF2E::mul_lut[r_raw][24]));
            const __m256i mask1 = _mm256_set1_epi8(0x0f);
            const __m256i mask2 = _mm256_set1_epi8(0xf0);

            for (int u = 0; u < (1 << (bit_len - i - 1)); u++) {
                int u0 = u << 1;
                int u1 = u0 + 1;
                // for (size_t p = 0; p < instance.num_MPC_parties; p++)
                //     inputs_rho_share[e][p][u] =
                //         inputs_rho_share[e][p][u<<1] +
                //         (inputs_rho_share[e][p][(u<<1)+1] - inputs_rho_share[e][p][u<<1]) * rho[i][e];
                for (size_t k = 0; k < instance.num_MPC_parties / 16; k++) {
                    __m256i lo, hi, tmp;
                    __m256i t_u0 = _mm256_loadu_si256((__m256i *)&(inputs_rho_share_v[e][u0][k * 32]));
                    __m256i t_u1 = _mm256_loadu_si256((__m256i *)&(inputs_rho_share_v[e][u1][k * 32]));
                    t_u1 = _mm256_xor_si256(t_u0, t_u1);

                    // (t_u1 + t_u0) * r
                    lo = _mm256_and_si256(t_u1, mask1);  // a0, a2
                    lo = _mm256_shuffle_epi8(table0, lo);   // a0lo, a2hi
                    hi = _mm256_srli_epi64(_mm256_and_si256(t_u1, mask2), 4);    // a1, a3
                    hi = _mm256_shuffle_epi8(table1, hi);   // a1lo, a3hi
                    tmp = _mm256_xor_si256(hi, lo);
                    t_u1 = _mm256_permute4x64_epi64(t_u1, 2 | (3 << 2) | (1 << 6));   // a2,a3; a0,a1
                    lo = _mm256_and_si256(t_u1, mask1);  // a2, a0
                    lo = _mm256_shuffle_epi8(table2, lo);   // a2lo, a0hi
                    hi = _mm256_srli_epi64(_mm256_and_si256(t_u1, mask2), 4);    // a3, a1
                    hi = _mm256_shuffle_epi8(table3, hi);   // a3lo, a1hi
                    t_u1 = _mm256_xor_si256(tmp, _mm256_xor_si256(hi, lo));

                    // + t_u0
                    t_u1 = _mm256_xor_si256(t_u0, t_u1);
                    _mm256_storeu_si256((__m256i *)&(inputs_rho_share_v[e][u][k * 32]), t_u1);
                }
            }

            for (size_t k = 0; k < instance.num_MPC_parties / 16; k++) {
                for (size_t p = 0; p < 16; p++) {
                    auto random_coef_share =
                        random_tapes.get_bytes(e, k * 16 + p,
                                instance.aes_params.key_size +
                                instance.aes_params.block_size * instance.aes_params.num_blocks +
                                instance.aes_params.num_sboxes +
                                3 * i * instance.lambda,
                                3 * instance.lambda);
                    coef_shares_v[e][i][0][k * 32 + p] = *(uint8_t *)(random_coef_share.data());
                    coef_shares_v[e][i][1][k * 32 + p] = *(uint8_t *)(random_coef_share.data() + 2);
                    coef_shares_v[e][i][2][k * 32 + p] = *(uint8_t *)(random_coef_share.data() + 4);
                    coef_shares_v[e][i][0][k * 32 + 16 + p] = *(uint8_t *)(random_coef_share.data() + 1);
                    coef_shares_v[e][i][1][k * 32 + 16 + p] = *(uint8_t *)(random_coef_share.data() + 3);
                    coef_shares_v[e][i][2][k * 32 + 16 + p] = *(uint8_t *)(random_coef_share.data() + 5);
                }
            }
            coef_shares_v[e][i][0][0] ^= (coef1_deltas[i][e][0].data & 255);
            coef_shares_v[e][i][1][0] ^= (coef1_deltas[i][e][1].data & 255);
            coef_shares_v[e][i][2][0] ^= (coef1_deltas[i][e][2].data & 255);
            coef_shares_v[e][i][0][16] ^= (coef1_deltas[i][e][0].data >> 8);
            coef_shares_v[e][i][1][16] ^= (coef1_deltas[i][e][1].data >> 8);
            coef_shares_v[e][i][2][16] ^= (coef1_deltas[i][e][2].data >> 8);
            for (size_t k = 0; k < instance.num_MPC_parties / 16; k++) {
                __m256i lo, hi, tmp;
                __m256i t_coef0 = _mm256_loadu_si256((__m256i *)&(coef_shares_v[e][i][0][k * 32]));
                __m256i t_coef1 = _mm256_loadu_si256((__m256i *)&(coef_shares_v[e][i][1][k * 32]));
                __m256i t_coef2 = _mm256_loadu_si256((__m256i *)&(coef_shares_v[e][i][2][k * 32]));
                __m256i t_fr    = _mm256_loadu_si256((__m256i *)&(fr_shares_v[e][k * 32]));
                t_fr = _mm256_xor_si256(_mm256_xor_si256(t_coef1, t_coef2), t_fr);
                // coef[2] * r
                //
                lo = _mm256_and_si256(t_coef2, mask1);  // a0, a2
                lo = _mm256_shuffle_epi8(table0, lo);   // a0lo, a2hi
                hi = _mm256_srli_epi64(_mm256_and_si256(t_coef2, mask2), 4);    // a1, a3
                hi = _mm256_shuffle_epi8(table1, hi);   // a1lo, a3hi
                tmp = _mm256_xor_si256(hi, lo);
                t_coef2 = _mm256_permute4x64_epi64(t_coef2, 2 | (3 << 2) | (1 << 6));   // a2,a3; a0,a1
                lo = _mm256_and_si256(t_coef2, mask1);  // a2, a0
                lo = _mm256_shuffle_epi8(table2, lo);   // a2lo, a0hi
                hi = _mm256_srli_epi64(_mm256_and_si256(t_coef2, mask2), 4);    // a3, a1
                hi = _mm256_shuffle_epi8(table3, hi);   // a3lo, a1hi
                t_coef2 = _mm256_xor_si256(tmp, _mm256_xor_si256(hi, lo));

                // + coef[1]
                t_coef2 = _mm256_xor_si256(t_coef1, t_coef2);
                // * r
                lo = _mm256_and_si256(t_coef2, mask1);  // a0, a2
                lo = _mm256_shuffle_epi8(table0, lo);   // a0lo, a2hi
                hi = _mm256_srli_epi64(_mm256_and_si256(t_coef2, mask2), 4);    // a1, a3
                hi = _mm256_shuffle_epi8(table1, hi);   // a1lo, a3hi
                tmp = _mm256_xor_si256(hi, lo);
                t_coef2 = _mm256_permute4x64_epi64(t_coef2, 2 | (3 << 2) | (1 << 6));   // a2,a3; a0,a1
                lo = _mm256_and_si256(t_coef2, mask1);  // a2, a0
                lo = _mm256_shuffle_epi8(table2, lo);   // a2lo, a0hi
                hi = _mm256_srli_epi64(_mm256_and_si256(t_coef2, mask2), 4);    // a3, a1
                hi = _mm256_shuffle_epi8(table3, hi);   // a3lo, a1hi
                t_coef2 = _mm256_xor_si256(tmp, _mm256_xor_si256(hi, lo));
                // + fr
                t_coef2 = _mm256_xor_si256(t_fr, t_coef2);
                // * r
                lo = _mm256_and_si256(t_coef2, mask1);  // a0, a2
                lo = _mm256_shuffle_epi8(table0, lo);   // a0lo, a2hi
                hi = _mm256_srli_epi64(_mm256_and_si256(t_coef2, mask2), 4);    // a1, a3
                hi = _mm256_shuffle_epi8(table1, hi);   // a1lo, a3hi
                tmp = _mm256_xor_si256(hi, lo);
                t_coef2 = _mm256_permute4x64_epi64(t_coef2, 2 | (3 << 2) | (1 << 6));   // a2,a3; a0,a1
                lo = _mm256_and_si256(t_coef2, mask1);  // a2, a0
                lo = _mm256_shuffle_epi8(table2, lo);   // a2lo, a0hi
                hi = _mm256_srli_epi64(_mm256_and_si256(t_coef2, mask2), 4);    // a3, a1
                hi = _mm256_shuffle_epi8(table3, hi);   // a3lo, a1hi
                t_coef2 = _mm256_xor_si256(tmp, _mm256_xor_si256(hi, lo));
                // + coef[0]
                t_coef2 = _mm256_xor_si256(t_coef0, t_coef2);

                _mm256_storeu_si256((__m256i *)&(fr_shares_v[e][k * 32]), t_coef2);
            }

            // for (size_t p = 0; p < instance.num_MPC_parties; p++) {
            //     auto shared_coef = coef1_shares.get(e, p);
            //     auto fr = fr_share.get(e, p);
            //     auto random_coef_share = random_tapes.get_bytes(e, p,
            //                 instance.aes_params.key_size +
            //                 instance.aes_params.num_sboxes +
            //                 3 * i * instance.lambda,
            //                 3 * instance.lambda);
            //     shared_coef[i][0].from_bytes(random_coef_share.data());
            //     shared_coef[i][1].from_bytes(random_coef_share.data() + instance.lambda);
            //     shared_coef[i][2].from_bytes(random_coef_share.data() + instance.lambda * 2);
            //     if (p == 0) {
            //         shared_coef[i][0] += coef1_deltas[i][e][0];
            //         shared_coef[i][1] += coef1_deltas[i][e][1];
            //         shared_coef[i][2] += coef1_deltas[i][e][2];
            //     }
            //     fr[0] -= shared_coef[i][1] + shared_coef[i][2];
            //     fr[0] = ((shared_coef[i][2] * rho[i][e] + shared_coef[i][1]) * rho[i][e] + fr[0]) * rho[i][e] + shared_coef[i][0];
            // }

        }
    for (int i = 0; i < bit_len; i++)
        for (size_t e = 0; e < instance.num_rounds; e++) {
            auto r = sgm[i][e];
            auto r_raw = r.data;
            __m256i table0 = _mm256_loadu_si256((__m256i *)&(field::GF2E::mul_lut[r_raw][0]));
            __m256i table1 = _mm256_loadu_si256((__m256i *)&(field::GF2E::mul_lut[r_raw][8]));
            __m256i table2 = _mm256_loadu_si256((__m256i *)&(field::GF2E::mul_lut[r_raw][16]));
            __m256i table3 = _mm256_loadu_si256((__m256i *)&(field::GF2E::mul_lut[r_raw][24]));
            __m256i mask1 = _mm256_set1_epi8(0x0f);
            __m256i mask2 = _mm256_set1_epi8(0xf0);
            for (int u = 0; u < (1 << (bit_len - i - 1)); u++) {
                int u0 = u << 1;
                int u1 = u0 + 1;
                // for (size_t p = 0; p < instance.num_MPC_parties; p++)
                //     inputs_sgm_share[e][p][u] =
                //         inputs_sgm_share[e][p][u<<1] +
                //         (inputs_sgm_share[e][p][(u<<1)+1] - inputs_sgm_share[e][p][u<<1]) * sgm[i][e];
                for (size_t k = 0; k < instance.num_MPC_parties / 16; k++) {
                    __m256i lo, hi, tmp;
                    __m256i t_u0 = _mm256_loadu_si256((__m256i *)&(inputs_sgm_share_v[e][u0][k * 32]));
                    __m256i t_u1 = _mm256_loadu_si256((__m256i *)&(inputs_sgm_share_v[e][u1][k * 32]));
                    t_u1 = _mm256_xor_si256(t_u0, t_u1);

                    // (t_u1 + t_u0) * r
                    lo = _mm256_and_si256(t_u1, mask1);  // a0, a2
                    lo = _mm256_shuffle_epi8(table0, lo);   // a0lo, a2hi
                    hi = _mm256_srli_epi64(_mm256_and_si256(t_u1, mask2), 4);    // a1, a3
                    hi = _mm256_shuffle_epi8(table1, hi);   // a1lo, a3hi
                    tmp = _mm256_xor_si256(hi, lo);
                    t_u1 = _mm256_permute4x64_epi64(t_u1, 2 | (3 << 2) | (1 << 6));   // a2,a3; a0,a1
                    lo = _mm256_and_si256(t_u1, mask1);  // a2, a0
                    lo = _mm256_shuffle_epi8(table2, lo);   // a2lo, a0hi
                    hi = _mm256_srli_epi64(_mm256_and_si256(t_u1, mask2), 4);    // a3, a1
                    hi = _mm256_shuffle_epi8(table3, hi);   // a3lo, a1hi
                    t_u1 = _mm256_xor_si256(tmp, _mm256_xor_si256(hi, lo));

                    // + t_u0
                    t_u1 = _mm256_xor_si256(t_u0, t_u1);
                    _mm256_storeu_si256((__m256i *)&(inputs_sgm_share_v[e][u][k * 32]), t_u1);
                }
            }

            for (size_t k = 0; k < instance.num_MPC_parties / 16; k++) {
                for (size_t p = 0; p < 16; p++) {
                    auto random_coef_share =
                        random_tapes.get_bytes(e, k * 16 + p,
                                instance.aes_params.key_size +
                                instance.aes_params.block_size * instance.aes_params.num_blocks +
                                instance.aes_params.num_sboxes +
                                3 * bit_len * instance.lambda +
                                2 * i * instance.lambda,
                                2 * instance.lambda);
                    coef_shares_v[e][i][3][k * 32 + p] = *(uint8_t *)(random_coef_share.data());
                    coef_shares_v[e][i][4][k * 32 + p] = *(uint8_t *)(random_coef_share.data() + 2);
                    coef_shares_v[e][i][3][k * 32 + 16 + p] = *(uint8_t *)(random_coef_share.data() + 1);
                    coef_shares_v[e][i][4][k * 32 + 16 + p] = *(uint8_t *)(random_coef_share.data() + 3);
                }
            }
            coef_shares_v[e][i][3][0] ^= (coef2_deltas[i][e][0].data & 255);
            coef_shares_v[e][i][4][0] ^= (coef2_deltas[i][e][1].data & 255);
            coef_shares_v[e][i][3][16] ^= (coef2_deltas[i][e][0].data >> 8);
            coef_shares_v[e][i][4][16] ^= (coef2_deltas[i][e][1].data >> 8);

            for (size_t k = 0; k < instance.num_MPC_parties / 16; k++) {
                __m256i lo, hi, tmp;
                __m256i t_coef0 = _mm256_loadu_si256((__m256i *)&(coef_shares_v[e][i][3][k * 32]));
                __m256i t_coef1 = _mm256_loadu_si256((__m256i *)&(coef_shares_v[e][i][4][k * 32]));
                __m256i t_fr    = _mm256_loadu_si256((__m256i *)&(fr_shares_v[e][k * 32]));
                t_fr = _mm256_xor_si256(t_coef1, t_fr);
                // coef1 * r
                lo = _mm256_and_si256(t_coef1, mask1);  // a0, a2
                lo = _mm256_shuffle_epi8(table0, lo);   // a0lo, a2hi
                hi = _mm256_srli_epi64(_mm256_and_si256(t_coef1, mask2), 4);    // a1, a3
                hi = _mm256_shuffle_epi8(table1, hi);   // a1lo, a3hi
                tmp = _mm256_xor_si256(hi, lo);
                t_coef1 = _mm256_permute4x64_epi64(t_coef1, 2 | (3 << 2) | (1 << 6));   // a2,a3; a0,a1
                lo = _mm256_and_si256(t_coef1, mask1);  // a2, a0
                lo = _mm256_shuffle_epi8(table2, lo);   // a2lo, a0hi
                hi = _mm256_srli_epi64(_mm256_and_si256(t_coef1, mask2), 4);    // a3, a1
                hi = _mm256_shuffle_epi8(table3, hi);   // a3lo, a1hi
                t_coef1 = _mm256_xor_si256(tmp, _mm256_xor_si256(hi, lo));
                // + fr
                t_coef1 = _mm256_xor_si256(t_fr, t_coef1);
                // * r
                lo = _mm256_and_si256(t_coef1, mask1);  // a0, a2
                lo = _mm256_shuffle_epi8(table0, lo);   // a0lo, a2hi
                hi = _mm256_srli_epi64(_mm256_and_si256(t_coef1, mask2), 4);    // a1, a3
                hi = _mm256_shuffle_epi8(table1, hi);   // a1lo, a3hi
                tmp = _mm256_xor_si256(hi, lo);
                t_coef1 = _mm256_permute4x64_epi64(t_coef1, 2 | (3 << 2) | (1 << 6));   // a2,a3; a0,a1
                lo = _mm256_and_si256(t_coef1, mask1);  // a2, a0
                lo = _mm256_shuffle_epi8(table2, lo);   // a2lo, a0hi
                hi = _mm256_srli_epi64(_mm256_and_si256(t_coef1, mask2), 4);    // a3, a1
                hi = _mm256_shuffle_epi8(table3, hi);   // a3lo, a1hi
                t_coef1 = _mm256_xor_si256(tmp, _mm256_xor_si256(hi, lo));
                // + coef[0]
                t_coef1 = _mm256_xor_si256(t_coef0, t_coef1);

                _mm256_storeu_si256((__m256i *)&(fr_shares_v[e][k * 32]), t_coef1);
            }

            // for (size_t p = 0; p < instance.num_MPC_parties; p++) {
            //     auto shared_coef = coef2_shares.get(e, p);
            //     auto fr = fr_share.get(e, p);
            //     auto random_coef_share = random_tapes.get_bytes(e, p,
            //                 instance.aes_params.key_size +
            //                 instance.aes_params.num_sboxes +
            //                 3 * bit_len * instance.lambda +
            //                 2 * i * instance.lambda,
            //                 2 * instance.lambda);
            //     shared_coef[i][0].from_bytes(random_coef_share.data());
            //     shared_coef[i][1].from_bytes(random_coef_share.data() + instance.lambda);
            //     if (p == 0) {
            //         shared_coef[i][0] += coef2_deltas[i][e][0];
            //         shared_coef[i][1] += coef2_deltas[i][e][1];
            //     }
            //     fr[0] -= shared_coef[i][1];
            //     fr[0] = (shared_coef[i][1] * sgm[i][e] + fr[0]) * sgm[i][e] + shared_coef[i][0];
            // }

        }
    for (size_t e = 0; e < instance.num_rounds; e++)
        for (size_t k = 0; k < instance.num_MPC_parties / 16; k++)
            for (size_t p = 0; p < 16; p++) {
                auto Vr = Vr_share.get(e, k * 16 + p);
                auto Vs = Vs_share.get(e, k * 16 + p);
                auto fr = fr_share.get(e, k * 16 + p);
                fr[0] = field::GF2E(fr_shares_v[e][k * 32 + p] | ((uint16_t)fr_shares_v[e][k * 32 + p + 16] << 8));
                Vr[0] = field::GF2E(inputs_rho_share_v[e][0][k * 32 + p] | ((uint16_t)inputs_rho_share_v[e][0][k * 32 + p + 16] << 8));
                Vs[0] = field::GF2E(inputs_sgm_share_v[e][0][k * 32 + p] | ((uint16_t)inputs_sgm_share_v[e][0][k * 32 + p + 16] << 8));
            }

    std::vector<field::GF2E> x_values_for_interpolation_zero_to_m2 = field::get_first_n_field_elements(1 + 1);
    std::vector<std::vector<field::GF2E>> precomputation_for_zero_to_m2 = precompute_lagrange_polynomials(x_values_for_interpolation_zero_to_m2);
    std::vector<field::GF2E> x_values_for_interpolation_zero_to_3m2 = field::get_first_n_field_elements(3 * 1 + 1);
    std::vector<std::vector<field::GF2E>> precomputation_for_zero_to_3m2 = precompute_lagrange_polynomials(x_values_for_interpolation_zero_to_3m2);

    std::vector<std::vector<std::vector<field::GF2E>>> fr_poly_share(instance.num_rounds);
    std::vector<std::vector<std::vector<field::GF2E>>> Vr_poly_share(instance.num_rounds);
    std::vector<std::vector<std::vector<field::GF2E>>> Vs_poly_share(instance.num_rounds);
    std::vector<std::vector<std::vector<field::GF2E>>> P_share(instance.num_rounds);
    for (size_t e = 0; e < instance.num_rounds; e++) {
        P_share[e].resize(instance.num_MPC_parties);
        fr_poly_share[e].resize(instance.num_MPC_parties);
        Vr_poly_share[e].resize(instance.num_MPC_parties);
        Vs_poly_share[e].resize(instance.num_MPC_parties);
        for (size_t p = 0; p < instance.num_MPC_parties; p++) {
            auto a = fr_share.get(e, p);
            auto b = Vr_share.get(e, p);
            auto c = Vs_share.get(e, p);
            auto random_share = random_tapes.get_bytes(e, p,
                        instance.aes_params.key_size +
                        instance.aes_params.block_size * instance.aes_params.num_blocks +
                        instance.aes_params.num_sboxes +
                        3 * bit_len * instance.lambda +
                        2 * bit_len * instance.lambda,
                        3 * instance.lambda + 3 * instance.lambda);
            a[1].from_bytes(random_share.data());
            b[1].from_bytes(random_share.data() + instance.lambda);
            c[1].from_bytes(random_share.data() + 2 * instance.lambda);
            fr_poly_share[e][p] = interpolate_with_precomputation(precomputation_for_zero_to_m2, a);
            Vr_poly_share[e][p] = interpolate_with_precomputation(precomputation_for_zero_to_m2, b);
            Vs_poly_share[e][p] = interpolate_with_precomputation(precomputation_for_zero_to_m2, c);
            P_share[e][p].resize(3 * 1 + 1);
            P_share[e][p][1].from_bytes(random_share.data() + 3 * instance.lambda);
            P_share[e][p][2].from_bytes(random_share.data() + 4 * instance.lambda);
            P_share[e][p][3].from_bytes(random_share.data() + 5 * instance.lambda);
            if (p == 0)
                for (size_t k = 1; k <= 3 * 1; k++)
                    P_share[e][0][k] += P_deltas[e][k - 1];
        }
    }

    /////////////////////////////////////////////////////////////////////////////
    // recompute views of polynomial checks
    /////////////////////////////////////////////////////////////////////////////

    std::vector<std::vector<field::GF2E>> a_shares(instance.num_rounds);
    std::vector<std::vector<field::GF2E>> b_shares(instance.num_rounds);
    std::vector<std::vector<field::GF2E>> c_shares(instance.num_rounds);
    std::vector<std::vector<field::GF2E>> d_shares(instance.num_rounds);

    std::vector<field::GF2E> lagrange_polys_evaluated_at_Re_m2(1 + 1);
    std::vector<field::GF2E> lagrange_polys_evaluated_at_Re_3m2(3 * 1 + 1);

    for (size_t e = 0; e < instance.num_rounds; e++) {
        field::GF2E a = signature.proofs[e].fr_at_R;
        field::GF2E b = signature.proofs[e].Vr_at_R;
        field::GF2E c = signature.proofs[e].Vs_at_R;
        field::GF2E d = a + pred[e] * b * (field::GF2E(1) - b * c);
        a_shares[e].resize(instance.num_MPC_parties);
        b_shares[e].resize(instance.num_MPC_parties);
        c_shares[e].resize(instance.num_MPC_parties);
        d_shares[e].resize(instance.num_MPC_parties);
        for (size_t k = 0; k < 1 + 1; k++)
            lagrange_polys_evaluated_at_Re_m2[k] = eval(precomputation_for_zero_to_m2[k], R_es[e]);
        for (size_t k = 0; k < 3 * 1 + 1; k++)
            lagrange_polys_evaluated_at_Re_3m2[k] = eval(precomputation_for_zero_to_3m2[k], R_es[e]);

        for (size_t p = 0; p < instance.num_MPC_parties; p++)
            if (p != missing_parties[e]) {
                auto fr   = fr_share.get(e, p);
                auto Vr = Vr_share.get(e, p);
                auto Vs = Vs_share.get(e, p);
                a_shares[e][p] = dot_product(lagrange_polys_evaluated_at_Re_m2, fr);
                b_shares[e][p] = dot_product(lagrange_polys_evaluated_at_Re_m2, Vr);
                c_shares[e][p] = dot_product(lagrange_polys_evaluated_at_Re_m2, Vs);
                d_shares[e][p] = dot_product(lagrange_polys_evaluated_at_Re_3m2, P_share[e][p]);
                a -= a_shares[e][p];
                b -= b_shares[e][p];
                c -= c_shares[e][p];
                d -= d_shares[e][p];
            }
        a_shares[e][missing_parties[e]] = a;
        b_shares[e][missing_parties[e]] = b;
        c_shares[e][missing_parties[e]] = c;
        d_shares[e][missing_parties[e]] = d;
    }

    std::vector<field::GF2E> a;
    std::vector<field::GF2E> b;
    std::vector<field::GF2E> c;
    for (const dubhe_repetition_group_proof_t &proof : signature.proofs) {
        a.push_back(proof.fr_at_R);
        b.push_back(proof.Vr_at_R);
        c.push_back(proof.Vs_at_R);
    }
    std::vector<uint8_t> h_3 = phase_3_commitment(instance, signature.salt, h_2, d_shares, a, a_shares, b, b_shares, c, c_shares);

    // group gkr sumcheck
    h_i = h_3;

    int random_offset = instance.aes_params.key_size +
                        instance.aes_params.block_size * instance.aes_params.num_blocks +
                        instance.aes_params.num_sboxes +
                        3 * bit_len * instance.lambda +
                        2 * bit_len * instance.lambda +
                        3 * instance.lambda + 3 * instance.lambda;
    std::vector<uint8_t> random_combinations = phase_group1_expand(instance, h_3, instance.num_rounds * gkey.size());
    std::vector<std::vector<std::vector<uint8_t>>> keytest_share_v(instance.num_rounds);
    for (size_t e = 0; e < instance.num_rounds; e++) {
        keytest_share_v[e].resize(gkey.size());
        for (size_t g = 0; g < gkey.size(); g++) {
            keytest_share_v[e][g].resize(instance.num_MPC_parties);
            int offset = (e * gkey.size() + g) * 2 * instance.aes_params.block_size * instance.aes_params.num_blocks;
            for (size_t k = 0; k < instance.num_MPC_parties / 16; k++) {
                __m256i res = _mm256_setzero_si256();
                for (unsigned int i = 0; i < instance.aes_params.block_size * instance.aes_params.num_blocks; i++) {
                    uint8_t t[32];
                    auto r0 = random_combinations[offset + i];
                    auto r1 = random_combinations[offset + i + instance.aes_params.block_size * instance.aes_params.num_blocks];
                    for (int p = 0; p < 16; p++) {
                        auto pt_shares = rep_shared_pt.get(e, k * 16 + p);
                        auto ct_shares = rep_shared_ct.get(e, k * 16 + p);
                        uint16_t pt, ct;
                        if (k == 0 && p == 0) {
                            pt = pt_shares[i] ^ gkey[g].second[i];
                            ct = ct_shares[i] ^ gkey[g].second[i + instance.aes_params.block_size * instance.aes_params.num_blocks];
                        } else {
                            pt = pt_shares[i];
                            ct = ct_shares[i];
                        }
                        t[p] = pt;
                        t[p + 16] = ct;
                    }
                    const __m256i mask1 = _mm256_set1_epi8(0x0f);
                    const __m256i mask2 = _mm256_set1_epi8(0xf0);
                    const __m256i table0 = _mm256_loadu_si256((__m256i *)&(gf256_mul_lut[r0 ^ (r1 << 8)][0]));
                    const __m256i table1 = _mm256_loadu_si256((__m256i *)&(gf256_mul_lut[r0 ^ (r1 << 8)][8]));
                    __m256i t_t = _mm256_loadu_si256((__m256i *)&t);

                    __m256i lo, hi;
                    lo = _mm256_and_si256(t_t, mask1);
                    lo = _mm256_shuffle_epi8(table0, lo);
                    hi = _mm256_and_si256(t_t, mask2);
                    hi = _mm256_srli_epi64(hi, 4);
                    hi = _mm256_shuffle_epi8(table1, hi);
                    t_t = _mm256_xor_si256(hi, lo);

                    res = _mm256_xor_si256(res, t_t);
                }
                _mm_storeu_si128((__m128i *)&(keytest_share_v[e][g][k * 16]),
                        _mm_xor_si128(
                            _mm256_castsi256_si128(res),
                            _mm256_extracti128_si256(res, 1)));
            }
        }
    }
    // RepContainer<std::array<field::GF2E, 2>> g_coef1_shares(instance.num_rounds, instance.num_MPC_parties, 1);
    // RepContainer<std::array<field::GF2E, 2>> g_coef2_shares(instance.num_rounds, instance.num_MPC_parties, 1);
    // RepContainer<field::GF2E> g_fr_share(instance.num_rounds, instance.num_MPC_parties, 2);
    std::vector<std::vector<std::vector<field::GF2E>>> g_Vr_shares(instance.num_rounds);
    std::vector<std::vector<std::vector<field::GF2E>>> g_Vs_shares(instance.num_rounds);
    std::vector<std::vector<std::vector<field::GF2E>>> g_fr_shares(instance.num_rounds);
    std::vector<std::vector<field::GF2E>> g_pred(instance.num_rounds);
    std::vector<std::vector<field::GF2E>> t_pred(instance.num_rounds);
    std::vector<std::vector<field::GF2E>> g_Vr_deltas(depth - 1);
    std::vector<std::vector<field::GF2E>> g_Vs_deltas(depth - 1);
    std::vector<std::vector<std::vector<std::vector<field::GF2E>>>> g_coef1_deltas(depth);
    std::vector<std::vector<std::vector<std::vector<field::GF2E>>>> g_coef2_deltas(depth);
    std::vector<std::vector<std::vector<std::array<std::vector<field::GF2E>, 4>>>> g_coef_shares(instance.num_rounds);
    std::vector<std::vector<std::vector<field::GF2E>>> g_rho(depth);
    std::vector<std::vector<std::vector<field::GF2E>>> g_sgm(depth);
    std::vector<std::vector<std::vector<field::GF2E>>> g_scalar(depth - 1);

    // init shares and pred
    for (size_t e = 0; e < instance.num_rounds; e++) {
        g_coef_shares[e].resize(depth);
        g_pred[e].resize(depth);
        t_pred[e].resize(depth);
        g_fr_shares[e].resize(instance.num_MPC_parties);
        g_Vr_shares[e].resize(instance.num_MPC_parties);
        g_Vs_shares[e].resize(instance.num_MPC_parties);
        for (int d = 0; d < depth; d++) {
            g_pred[e][d] = field::GF2E(1);
            t_pred[e][d] = field::GF2E(1);
            g_coef_shares[e][d].resize(d + 1);
            for (int i = 0; i < d + 1; i++)
                for (int j = 0; j < 4; j++)
                    g_coef_shares[e][d][i][j].resize(instance.num_MPC_parties);
        }
        for (size_t p = 0; p < instance.num_MPC_parties; p++) {
            g_fr_shares[e][p].resize(depth + 1);
            g_Vr_shares[e][p].resize(depth + 1);
            g_Vs_shares[e][p].resize(depth + 1);
        }
    }
    // retrieve deltas from proof
    // compute rho, sgm, scalar and pred
    field::GF2E one(1);
    for (int d = 0; d < depth; d++) {
        if (d != 0) {
            g_Vr_deltas[d-1].resize(instance.num_rounds);
            g_Vs_deltas[d-1].resize(instance.num_rounds);
            for (size_t e = 0; e < instance.num_rounds; e++) {
                g_Vr_deltas[d-1][e] = signature.proofs[e].g_Vr_delta[d-1];
                g_Vs_deltas[d-1][e] = signature.proofs[e].g_Vs_delta[d-1];
            }
            h_i = phase_sumcheck_final_commitment(instance, signature.salt, h_i, g_Vr_deltas[d-1], g_Vs_deltas[d-1]);
            std::vector<std::vector<field::GF2E>> scalar_d = phase_sumcheck_final_expand(instance, h_i);
            g_scalar[d-1] = scalar_d;
        }
        g_coef1_deltas[d].resize(d + 1);
        g_coef2_deltas[d].resize(d + 1);
        for (int i = 0; i < d + 1; i++) {
            g_coef1_deltas[d][i].resize(instance.num_rounds);
            for (size_t e = 0; e < instance.num_rounds; e++) {
                g_coef1_deltas[d][i][e] = signature.proofs[e].g_coef1_delta[d][i];    // vector of 2 coefs
            }
            h_i = phase_sumcheck_commitment(instance, signature.salt, h_i, g_coef1_deltas[d][i]);
            std::vector<field::GF2E> rho_i = phase_sumcheck_expand(instance, h_i);
            g_rho[d].push_back(rho_i);
        }
        for (int i = 0; i < d + 1; i++) {
            g_coef2_deltas[d][i].resize(instance.num_rounds);
            for (size_t e = 0; e < instance.num_rounds; e++) {
                g_coef2_deltas[d][i][e] = signature.proofs[e].g_coef2_delta[d][i];    // vector of 2 coefs
            }
            h_i = phase_sumcheck_commitment(instance, signature.salt, h_i, g_coef2_deltas[d][i]);
            std::vector<field::GF2E> sgm_i = phase_sumcheck_expand(instance, h_i);
            g_sgm[d].push_back(sgm_i);
            for (size_t e = 0; e < instance.num_rounds; e++) {
                if (d == 0)
                    g_pred[e][d] *= (one + g_rho[d][i][e]) * g_sgm[d][i][e];
                else if (i == 0) {
                    g_pred[e][d] *= g_scalar[d-1][e][0] * (one + g_rho[d][i][e]) * g_sgm[d][i][e];
                    t_pred[e][d] *= g_scalar[d-1][e][1] * (one + g_rho[d][i][e]) * g_sgm[d][i][e];
                } else {
                    g_pred[e][d] *= g_rho[d][i][e] * g_sgm[d][i][e] * g_rho[d-1][i-1][e] + (one + g_rho[d][i][e]) * (one + g_sgm[d][i][e]) * (one + g_rho[d-1][i-1][e]);
                    t_pred[e][d] *= g_rho[d][i][e] * g_sgm[d][i][e] * g_sgm[d-1][i-1][e] + (one + g_rho[d][i][e]) * (one + g_sgm[d][i][e]) * (one + g_sgm[d-1][i-1][e]);
                }
            }
        }
        if (d > 0)
            for (size_t e = 0; e < instance.num_rounds; e++)
                g_pred[e][d] += t_pred[e][d];
    }

    // loop over layers, compute shares of Vr, Vs, fr
    for (int d = 0; d < depth; d++) {
        if (d != 0) {
            for (size_t e = 0; e < instance.num_rounds; e++) {
                for (size_t p = 0; p < instance.num_MPC_parties; p++) {
                    auto random_coef_share =
                        random_tapes.get_bytes(e, p,
                                random_offset,
                                2 * instance.lambda);
                    g_Vr_shares[e][p][d-1].from_bytes(random_coef_share.data());
                    g_Vs_shares[e][p][d-1].from_bytes(random_coef_share.data() + instance.lambda);
                    if (p == 0) {
                        g_Vr_shares[e][0][d-1] += g_Vr_deltas[d-1][e];
                        g_Vs_shares[e][0][d-1] += g_Vs_deltas[d-1][e];
                    }
                }
            }
            random_offset += 2 * instance.lambda;
            for (size_t e = 0; e < instance.num_rounds; e++) {
                for (size_t p = 0; p < instance.num_MPC_parties; p++) {
                    g_fr_shares[e][p][d] = g_scalar[d-1][e][0] * g_Vr_shares[e][p][d-1] + g_scalar[d-1][e][1] * g_Vs_shares[e][p][d-1];
                }
            }
        }
        // layer d phase 1
        for (int i = 0; i < d + 1; i++) {
            for (size_t e = 0; e < instance.num_rounds; e++) {
                for (size_t p = 0; p < instance.num_MPC_parties; p++) {
                    auto random_coef_share = random_tapes.get_bytes(e, p,
                            random_offset,
                            2 * instance.lambda);
                    g_coef_shares[e][d][i][0][p].from_bytes(random_coef_share.data());
                    g_coef_shares[e][d][i][1][p].from_bytes(random_coef_share.data() + instance.lambda);
                    if (p == 0) {
                        g_coef_shares[e][d][i][0][0] += g_coef1_deltas[d][i][e][0];
                        g_coef_shares[e][d][i][1][0] += g_coef1_deltas[d][i][e][1];
                    }
                    g_fr_shares[e][p][d] -= g_coef_shares[e][d][i][1][p];
                    g_fr_shares[e][p][d] = (g_coef_shares[e][d][i][1][p] * g_rho[d][i][e] + g_fr_shares[e][p][d]) * g_rho[d][i][e] + g_coef_shares[e][d][i][0][p];
                }
            }
            random_offset += 2 * instance.lambda;
        }
        for (int i = 0; i < d + 1; i++) {
            for (size_t e = 0; e < instance.num_rounds; e++) {
                for (size_t p = 0; p < instance.num_MPC_parties; p++) {
                    auto random_coef_share = random_tapes.get_bytes(e, p,
                            random_offset,
                            2 * instance.lambda);
                    g_coef_shares[e][d][i][2][p].from_bytes(random_coef_share.data());
                    g_coef_shares[e][d][i][3][p].from_bytes(random_coef_share.data() + instance.lambda);
                    if (p == 0) {
                        g_coef_shares[e][d][i][2][0] += g_coef2_deltas[d][i][e][0];
                        g_coef_shares[e][d][i][3][0] += g_coef2_deltas[d][i][e][1];
                    }
                    g_fr_shares[e][p][d] -= g_coef_shares[e][d][i][3][p];
                    g_fr_shares[e][p][d] = (g_coef_shares[e][d][i][3][p] * g_sgm[d][i][e] + g_fr_shares[e][p][d]) * g_sgm[d][i][e] + g_coef_shares[e][d][i][2][p];
                }
            }
            random_offset += 2 * instance.lambda;
        }
    }

    // input layer Vr and Vs shares computed from keytest shares
    for (size_t e = 0; e < instance.num_rounds; e++) {
        // std::vector<std::vector<field::GF2E>> tmpr = keytest_share[e];
        // std::vector<std::vector<field::GF2E>> tmps = keytest_share[e];
        std::vector<std::vector<field::GF2E>> tmpr(gkey.size());
        std::vector<std::vector<field::GF2E>> tmps(gkey.size());
        for (size_t g = 0; g < gkey.size(); g++) {
            tmpr[g].resize(instance.num_MPC_parties);
            tmps[g].resize(instance.num_MPC_parties);
            for (size_t k = 0; k < instance.num_MPC_parties / 16; k++) {
                for (size_t p = 0; p < 16; p++) {
                    auto tmp = field::lift_uint8_t(keytest_share_v[e][g][k * 16 + p]);
                    tmpr[g][k * 16 + p] = tmp;
                    tmps[g][k * 16 + p] = tmp;
                }
            }
        }
        for (int i = 0; i < depth; i++) {
            int mask = 1 << (depth - i - 1);
            for (int u = 0; u < mask; u++) {
                int u0 = u << 1;
                int u1 = u0 + 1;
                for (size_t p = 0; p < instance.num_MPC_parties; p++) {
                    tmpr[u][p] = tmpr[u0][p] + (tmpr[u0][p] + tmpr[u1][p]) * g_rho[depth-1][i][e];
                    tmps[u][p] = tmps[u0][p] + (tmps[u0][p] + tmps[u1][p]) * g_sgm[depth-1][i][e];
                }
            }
        }
        for (size_t p = 0; p < instance.num_MPC_parties; p++) {
            g_Vr_shares[e][p][depth-1] = tmpr[0][p];
            g_Vs_shares[e][p][depth-1] = tmps[0][p];
        }
    }

    // merge g_pred into g_Vs_shares
    for (size_t e = 0; e < instance.num_rounds; e++)
        for (int d = 0; d < depth; d++)
            for (size_t p = 0; p < instance.num_MPC_parties; p++)
                g_Vs_shares[e][p][d] *= g_pred[e][d];
    // prove that sum(g_Vr_shares) * sum(g_Vs_shares) == sum(g_fr_share)

    // polynomials
    std::vector<field::GF2E> x_values_for_interpolation_zero_to_d = field::get_first_n_field_elements(depth + 1);
    std::vector<std::vector<field::GF2E>> precomputation_for_zero_to_d = field::precompute_lagrange_polynomials(x_values_for_interpolation_zero_to_d);
    std::vector<field::GF2E> x_values_for_interpolation_zero_to_2d = field::get_first_n_field_elements(2 * depth + 1);
    std::vector<std::vector<field::GF2E>> precomputation_for_zero_to_2d = field::precompute_lagrange_polynomials(x_values_for_interpolation_zero_to_2d);

    std::vector<std::vector<std::vector<field::GF2E>>> g_fr_poly_share(instance.num_rounds);
    std::vector<std::vector<std::vector<field::GF2E>>> g_Vr_poly_share(instance.num_rounds);
    std::vector<std::vector<std::vector<field::GF2E>>> g_Vs_poly_share(instance.num_rounds);
    std::vector<std::vector<std::vector<field::GF2E>>> g_P_share(instance.num_rounds);
    std::vector<std::vector<field::GF2E>> g_P_deltas(instance.num_rounds);
    for (size_t e = 0; e < instance.num_rounds; e++) {
        g_P_deltas[e] = signature.proofs[e].g_P_delta;
        g_P_share[e].resize(instance.num_MPC_parties);
        g_fr_poly_share[e].resize(instance.num_MPC_parties);
        g_Vr_poly_share[e].resize(instance.num_MPC_parties);
        g_Vs_poly_share[e].resize(instance.num_MPC_parties);
        for (size_t p = 0; p < instance.num_MPC_parties; p++) {
            auto random_share = random_tapes.get_bytes(e, p,
                        random_offset,
                        (4 + depth) * instance.lambda);
            g_fr_shares[e][p][depth].from_bytes(random_share.data());
            g_Vr_shares[e][p][depth].from_bytes(random_share.data() + instance.lambda);
            g_Vs_shares[e][p][depth].from_bytes(random_share.data() + 2 * instance.lambda);

            g_fr_poly_share[e][p] = interpolate_with_precomputation(precomputation_for_zero_to_d, g_fr_shares[e][p]);
            g_Vr_poly_share[e][p] = interpolate_with_precomputation(precomputation_for_zero_to_d, g_Vr_shares[e][p]);
            g_Vs_poly_share[e][p] = interpolate_with_precomputation(precomputation_for_zero_to_d, g_Vs_shares[e][p]);

            g_P_share[e][p].resize(2 * depth + 1);
            for (int i = depth; i < 2 * depth + 1; i++) {
                g_P_share[e][p][i].from_bytes(random_share.data() + (3 + i - depth) * instance.lambda);
            }
            P_share[e][p][1].from_bytes(random_share.data() + 3 * instance.lambda);
            P_share[e][p][2].from_bytes(random_share.data() + 4 * instance.lambda);
            P_share[e][p][3].from_bytes(random_share.data() + 5 * instance.lambda);
            if (p == 0)
                for (int i = depth; i < 2 * depth + 1; i++)
                    g_P_share[e][0][i] += g_P_deltas[e][i - depth];
        }
    }
    std::vector<uint8_t> h_4 = phase_2_commitment(instance, signature.salt, h_i, g_P_deltas);
    // expand challenge hash to depth values
    std::vector<field::GF2E> g_forbidden_challenge_values = field::get_first_n_field_elements(depth);
    std::vector<field::GF2E> g_R_es = phase_2_expand(instance, h_4, g_forbidden_challenge_values);

    // recompute views of polynomial checks
    std::vector<std::vector<field::GF2E>> g_a_shares(instance.num_rounds);
    std::vector<std::vector<field::GF2E>> g_b_shares(instance.num_rounds);
    std::vector<std::vector<field::GF2E>> g_c_shares(instance.num_rounds);
    std::vector<std::vector<field::GF2E>> g_d_shares(instance.num_rounds);
    std::vector<field::GF2E> lagrange_polys_evaluated_at_Re_d(depth + 1);
    std::vector<field::GF2E> lagrange_polys_evaluated_at_Re_2d(2 * depth + 1);

    for (size_t e = 0; e < instance.num_rounds; e++) {
        field::GF2E g_a = signature.proofs[e].g_fr_at_R;
        field::GF2E g_b = signature.proofs[e].g_Vr_at_R;
        field::GF2E g_c = signature.proofs[e].g_Vs_at_R;
        field::GF2E g_d = g_a + g_b * g_c;
        g_a_shares[e].resize(instance.num_MPC_parties);
        g_b_shares[e].resize(instance.num_MPC_parties);
        g_c_shares[e].resize(instance.num_MPC_parties);
        g_d_shares[e].resize(instance.num_MPC_parties);
        for (int k = 0; k < depth + 1; k++)
            lagrange_polys_evaluated_at_Re_d[k] = field::eval(precomputation_for_zero_to_d[k], g_R_es[e]);
        for (int k = 0; k < 2 * depth + 1; k++)
            lagrange_polys_evaluated_at_Re_2d[k] = field::eval(precomputation_for_zero_to_2d[k], g_R_es[e]);

        for (size_t p = 0; p < instance.num_MPC_parties; p++)
            if (p != missing_parties[e]) {
                g_a_shares[e][p] = dot_product(lagrange_polys_evaluated_at_Re_d, g_fr_shares[e][p]);
                g_b_shares[e][p] = dot_product(lagrange_polys_evaluated_at_Re_d, g_Vr_shares[e][p]);
                g_c_shares[e][p] = dot_product(lagrange_polys_evaluated_at_Re_d, g_Vs_shares[e][p]);
                g_d_shares[e][p] = dot_product(lagrange_polys_evaluated_at_Re_2d, g_P_share[e][p]);
                g_a -= g_a_shares[e][p];
                g_b -= g_b_shares[e][p];
                g_c -= g_c_shares[e][p];
                g_d -= g_d_shares[e][p];
            }
        g_a_shares[e][missing_parties[e]] = g_a;
        g_b_shares[e][missing_parties[e]] = g_b;
        g_c_shares[e][missing_parties[e]] = g_c;
        g_d_shares[e][missing_parties[e]] = g_d;
    }

    /////////////////////////////////////////////////////////////////////////////
    // recompute h_1 and h_5
    /////////////////////////////////////////////////////////////////////////////

    std::vector<std::vector<uint8_t>> sk_deltas;
    std::vector<std::vector<uint8_t>> t_deltas;
    std::vector<std::vector<uint8_t>> pk_deltas;
    for (const dubhe_repetition_group_proof_t &proof : signature.proofs) {
        sk_deltas.push_back(proof.sk_delta);
        t_deltas.push_back(proof.t_delta);
        pk_deltas.push_back(proof.pk_delta);
    }

    std::vector<uint8_t> h_1 = group_phase_1_commitment(
            instance, signature.salt, message, message_len,
            party_seed_commitments, sk_deltas, t_deltas, pk_deltas);

    std::vector<field::GF2E> g_a(instance.num_rounds);
    std::vector<field::GF2E> g_b(instance.num_rounds);
    std::vector<field::GF2E> g_c(instance.num_rounds);
    for (size_t e = 0; e < instance.num_rounds; e++) {
        g_a[e] = signature.proofs[e].g_fr_at_R;
        g_b[e] = signature.proofs[e].g_Vr_at_R;
        g_c[e] = signature.proofs[e].g_Vs_at_R;
    }
    std::vector<uint8_t> h_5 = phase_3_commitment(instance, signature.salt, h_4, g_d_shares, g_a, g_a_shares, g_b, g_b_shares, g_c, g_c_shares);

    if (memcmp(h_1.data(), signature.h_1.data(), h_1.size()) != 0) {
        return false;
    }
    if (memcmp(h_5.data(), signature.h_5.data(), h_5.size()) != 0) {
        return false;
    }

    return true;
}

std::vector<uint8_t>
dubhe_serialize_group_signature(const dubhe_instance_t &instance,
        size_t gsize, const dubhe_group_signature_t &signature) {

    std::vector<uint8_t> serialized;

    serialized.insert(serialized.end(), signature.salt.begin(),
            signature.salt.end());
    serialized.insert(serialized.end(), signature.h_1.begin(),
            signature.h_1.end());
    serialized.insert(serialized.end(), signature.h_5.begin(),
            signature.h_5.end());

    int depth = ceil_log2(gsize);

    for (size_t repetition = 0; repetition < instance.num_rounds; repetition++) {
        const dubhe_repetition_group_proof_t &proof = signature.proofs[repetition];
        for (const std::vector<uint8_t> &seed : proof.reveallist.first) {
            serialized.insert(serialized.end(), seed.begin(), seed.end());
        }
        serialized.insert(serialized.end(), proof.C_e.begin(), proof.C_e.end());
        serialized.insert(serialized.end(), proof.sk_delta.begin(),
                proof.sk_delta.end());
        serialized.insert(serialized.end(), proof.pk_delta.begin(),
                proof.pk_delta.end());
        serialized.insert(serialized.end(), proof.t_delta.begin(),
                proof.t_delta.end());
        for (size_t k = 0; k < instance.aes_params.bit_len; k++) {
            for (int i = 0; i < 3; i++) {
                std::vector<uint8_t> buffer = proof.coef1_delta[k][i].to_bytes();
                serialized.insert(serialized.end(), buffer.begin(), buffer.end());
            }
        }
        for (size_t k = 0; k < instance.aes_params.bit_len; k++) {
            for (int i = 0; i < 2; i++) {
                std::vector<uint8_t> buffer = proof.coef2_delta[k][i].to_bytes();
                serialized.insert(serialized.end(), buffer.begin(), buffer.end());
            }
        }

        for (size_t k = 0; k < 2 * 1 + 1; k++) {
            std::vector<uint8_t> buffer = proof.P_delta[k].to_bytes();
            serialized.insert(serialized.end(), buffer.begin(), buffer.end());
        }
        {
            std::vector<uint8_t> buffer = proof.fr_at_R.to_bytes();
            serialized.insert(serialized.end(), buffer.begin(), buffer.end());
        }
        {
            std::vector<uint8_t> buffer = proof.Vr_at_R.to_bytes();
            serialized.insert(serialized.end(), buffer.begin(), buffer.end());
        }
        {
            std::vector<uint8_t> buffer = proof.Vs_at_R.to_bytes();
            serialized.insert(serialized.end(), buffer.begin(), buffer.end());
        }

        for (int d = 0; d < depth; d++)
            for (int i = 0; i < d + 1; i++)
                for (int j = 0; j < 2; j++) {
                    std::vector<uint8_t> buffer = proof.g_coef1_delta[d][i][j].to_bytes();
                    serialized.insert(serialized.end(), buffer.begin(), buffer.end());
                }
        for (int d = 0; d < depth; d++)
            for (int i = 0; i < d + 1; i++)
                for (int j = 0; j < 2; j++) {
                    std::vector<uint8_t> buffer = proof.g_coef2_delta[d][i][j].to_bytes();
                    serialized.insert(serialized.end(), buffer.begin(), buffer.end());
                }
        for (int d = 0; d < depth - 1; d++) {
            std::vector<uint8_t> buffer = proof.g_Vr_delta[d].to_bytes();
            serialized.insert(serialized.end(), buffer.begin(), buffer.end());
        }
        for (int d = 0; d < depth - 1; d++) {
            std::vector<uint8_t> buffer = proof.g_Vs_delta[d].to_bytes();
            serialized.insert(serialized.end(), buffer.begin(), buffer.end());
        }
        for (int d = 0; d < depth + 1; d++) {
            std::vector<uint8_t> buffer = proof.g_P_delta[d].to_bytes();
            serialized.insert(serialized.end(), buffer.begin(), buffer.end());
        }
        {
            std::vector<uint8_t> buffer = proof.g_fr_at_R.to_bytes();
            serialized.insert(serialized.end(), buffer.begin(), buffer.end());
        }
        {
            std::vector<uint8_t> buffer = proof.g_Vr_at_R.to_bytes();
            serialized.insert(serialized.end(), buffer.begin(), buffer.end());
        }
        {
            std::vector<uint8_t> buffer = proof.g_Vs_at_R.to_bytes();
            serialized.insert(serialized.end(), buffer.begin(), buffer.end());
        }
    }
    return serialized;
}
*/
