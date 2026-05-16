//! Image cache with LRU eviction — deduplicates decoded GPU textures by file path.

use slint::Image;
use std::collections::{HashMap, VecDeque};

pub struct ImageCache {
    images: HashMap<String, Image>,
    order: VecDeque<String>,
    max_size: usize,
}

impl ImageCache {
    pub fn new(max_size: usize) -> Self {
        Self {
            images: HashMap::new(),
            order: VecDeque::new(),
            max_size,
        }
    }

    /// Get or load an image from the given path. Returns Image::default() on failures.
    pub fn get_or_load(&mut self, path: &str) -> Image {
        if path.is_empty() {
            return Image::default();
        }
        // Touch: move to back of LRU order
        if let Some(img) = self.images.get(path) {
            if let Some(pos) = self.order.iter().position(|p| p == path) {
                self.order.remove(pos);
            }
            self.order.push_back(path.to_string());
            return img.clone();
        }
        // Load from disk
        let img = Image::load_from_path(std::path::Path::new(path)).unwrap_or_default();
        // Evict if at capacity
        while self.images.len() >= self.max_size {
            if let Some(old) = self.order.pop_front() {
                self.images.remove(&old);
            } else {
                break;
            }
        }
        self.images.insert(path.to_string(), img.clone());
        self.order.push_back(path.to_string());
        img
    }

    /// Release all cached textures.
    pub fn clear(&mut self) {
        self.images.clear();
        self.order.clear();
    }

    /// Number of cached images.
    pub fn len(&self) -> usize {
        self.images.len()
    }
}
