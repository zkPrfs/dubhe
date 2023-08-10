#pragma once

#include <array>
#include <cstdint>
#include <cstdlib>
#include <vector>

#include "dubhe_instances.h"
#include "types.h"

// crypto api
dubhe_keypair_t dubhe_keygen(const dubhe_instance_t &instance);

dubhe_signature_t dubhe_sign(const dubhe_instance_t &instance,
                                 const dubhe_keypair_t &keypair,
                                 const uint8_t *message, size_t message_len);

dubhe_group_signature_t dubhe_group_sign(const dubhe_instance_t &instance,
                                 const dubhe_keypair_t &keypair,
                                 const std::vector<dubhe_keypair_t> &gkey,
                                 const uint8_t *message, size_t message_len);

bool dubhe_verify(const dubhe_instance_t &instance,
                    const std::vector<uint8_t> &pk,
                    const dubhe_signature_t &signature,
                    const uint8_t *message, size_t message_len);

bool dubhe_group_verify(const dubhe_instance_t &instance,
                          const dubhe_group_signature_t &signature,
                          const std::vector<dubhe_keypair_t> &gkey,
                          const uint8_t *message, size_t message_len);

std::vector<uint8_t>
dubhe_serialize_signature(const dubhe_instance_t &instance,
                            const dubhe_signature_t &signature);
dubhe_signature_t
dubhe_deserialize_signature(const dubhe_instance_t &instance,
                              const std::vector<uint8_t> &signature);
std::vector<uint8_t>
dubhe_serialize_group_signature(const dubhe_instance_t &instance,
                                  size_t gsize,
                                  const dubhe_group_signature_t &signature);
dubhe_group_signature_t
dubhe_deserialize_group_signature(const dubhe_instance_t &instance,
                                    const std::vector<uint8_t> &signature);
