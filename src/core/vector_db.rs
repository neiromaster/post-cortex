// Copyright (c) 2025 Julius ML
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//! Fast Vector Database with HNSW for semantic similarity search - NOW LOCK-FREE!
//!
//! This module has been refactored to use lock-free implementation internally.
//! The API remains the same for backward compatibility.
//!
//! The implementation now uses:
//! - DashMap for lock-free concurrent hash map operations
//! - Atomic operations for statistics
//! - ArcSwap for quantization parameters
//! - No blocking synchronization primitives

// Re-export lock-free types for backward compatibility
pub use crate::core::lockfree_vector_db::{
    LockFreeVectorDB as FastVectorDB,
    LockFreeVectorDbConfig as VectorDbConfig,
    VectorMetadata,
    SearchMatch,
    VectorDbStatsSnapshot as VectorDbStats,
};