use std::fmt;

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Key {
    pub bytes: Vec<u8>,
}

impl From<&str> for Key {
    fn from(value: &str) -> Self {
        Key {
            bytes: value.bytes().collect(),
        }
    }
}

impl From<String> for Key {
    fn from(value: String) -> Self {
        Key {
            bytes: value.into_bytes(),
        }
    }
}

impl From<usize> for Key {
    fn from(value: usize) -> Self {
        // By storing the values as big endian bytes we can compare them in
        // lexicographic order.
        Key {
            bytes: value.to_be_bytes().to_vec(),
        }
    }
}

impl fmt::Display for Key {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let number = if self.bytes.len() == 8 {
            Some(usize::from_be_bytes(self.bytes.clone().try_into().unwrap()))
        } else {
            None
        };
        let text = String::from_utf8(self.bytes.clone()).ok();
        match (number, text) {
            (None, None) => write!(f, "{:?}", self.bytes),
            (None, Some(text)) => write!(f, "\"{text}\""),
            (Some(number), None) => write!(f, "{number}"),
            (Some(number), Some(_)) => write!(f, "{number}"),
        }
    }
}

impl fmt::Debug for Key {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self.bytes)?;
        if let Ok(s) = str::from_utf8(&self.bytes) {
            write!(f, " ({s})")?;
        }
        Ok(())
    }
}
