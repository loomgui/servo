/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

//! A fast path for the color strings canvas pages set most often.
//!
//! Pages that animate a canvas build an `rgba(r, g, b, a)` string for nearly every fill and
//! gradient stop, with an alpha that changes every frame. The CSS parser takes several hundred
//! nanoseconds for each one. [`parse_legacy_rgb_color`] recognizes the plainest form of that
//! syntax directly and declines everything else, so callers fall back to the CSS parser.

use style::color::AbsoluteColor;

/// Parse `rgb(r, g, b)` or `rgba(r, g, b, a)` written the plain way:
///
/// - lowercase function name, comma-separated arguments, optional spaces around them;
/// - `r`, `g` and `b` as unsigned integers no greater than 255;
/// - `a`, if present, as an unsigned decimal number without an exponent or a percent sign.
///
/// Returns `None` for any other input, including valid colors in another form. For the input it
/// accepts, the result is bit-for-bit the color the CSS parser computes.
pub fn parse_legacy_rgb_color(input: &str) -> Option<AbsoluteColor> {
    let bytes = input.as_bytes();
    let arguments = bytes
        .strip_prefix(b"rgba(")
        .or_else(|| bytes.strip_prefix(b"rgb("))?;
    let mut cursor = Cursor(arguments);

    let red = cursor.channel()?;
    cursor.separator(b',')?;
    let green = cursor.channel()?;
    cursor.separator(b',')?;
    let blue = cursor.channel()?;
    cursor.skip_spaces();
    let alpha = if cursor.peek() == Some(b',') {
        cursor.advance();
        cursor.alpha()?
    } else {
        1.0
    };
    cursor.separator(b')')?;
    if !cursor.0.is_empty() {
        return None;
    }

    Some(AbsoluteColor::srgb_legacy(red, green, blue, alpha))
}

struct Cursor<'a>(&'a [u8]);

impl Cursor<'_> {
    fn peek(&self) -> Option<u8> {
        self.0.first().copied()
    }

    fn advance(&mut self) {
        self.0 = &self.0[1..];
    }

    fn skip_spaces(&mut self) {
        while self.peek() == Some(b' ') {
            self.advance();
        }
    }

    fn digit(&mut self) -> Option<u8> {
        let digit = self.peek().filter(u8::is_ascii_digit)? - b'0';
        self.advance();
        Some(digit)
    }

    /// Skip spaces, then require `byte`.
    fn separator(&mut self, byte: u8) -> Option<()> {
        self.skip_spaces();
        if self.peek()? != byte {
            return None;
        }
        self.advance();
        Some(())
    }

    /// An unsigned integer no greater than 255. A value followed by anything other than a space
    /// or a separator (a fraction, an exponent, a percent sign) is left for the CSS parser by the
    /// caller's next `separator` call.
    fn channel(&mut self) -> Option<u8> {
        self.skip_spaces();
        let mut value = u32::from(self.digit()?);
        while let Some(digit) = self.digit() {
            value = value * 10 + u32::from(digit);
            if value > 255 {
                return None;
            }
        }
        Some(value as u8)
    }

    /// An unsigned decimal number, clamped to [0, 1].
    ///
    /// The digits are accumulated exactly the way `cssparser`'s tokenizer does it (in `f64`, the
    /// fraction as a running sum of `digit * 10^-n`), so the `f32` that comes out is the same one
    /// the CSS parser would produce, not merely a close one.
    fn alpha(&mut self) -> Option<f32> {
        self.skip_spaces();
        let mut integral_part = f64::from(self.digit()?);
        while let Some(digit) = self.digit() {
            integral_part = integral_part * 10. + f64::from(digit);
        }
        let mut fractional_part: f64 = 0.;
        if self.peek() == Some(b'.') {
            self.advance();
            let mut factor = 0.1;
            fractional_part += f64::from(self.digit()?) * factor;
            factor *= 0.1;
            while let Some(digit) = self.digit() {
                fractional_part += f64::from(digit) * factor;
                factor *= 0.1;
            }
        }
        let value = (integral_part + fractional_part) as f32;
        Some(value.clamp(0.0, 1.0))
    }
}

#[cfg(test)]
mod tests {
    use cssparser::Parser;
    use style::color::AbsoluteColor;
    use style::context::QuirksMode;
    use style::parser::ParserContext;
    use style::stylesheets::{CssRuleType, Origin, UrlExtraData};
    use style::values::computed::color::Color as ComputedColor;
    use style::values::specified::color::Color;
    use style_traits::ParsingMode;

    use super::parse_legacy_rgb_color;

    /// What the canvas API computes for a color string without the fast path.
    fn css_parser_color(input: &str) -> Option<AbsoluteColor> {
        let url_data: UrlExtraData = url::Url::parse("about:blank").unwrap().into();
        let context = ParserContext::new(
            Origin::Author,
            &url_data,
            Some(CssRuleType::Style),
            ParsingMode::DEFAULT,
            QuirksMode::NoQuirks,
            Default::default(),
            None,
            None,
            Default::default(),
        );
        let mut parser = Parser::new(input);
        match Color::parse_and_compute(&context, &mut parser, None) {
            Ok(ComputedColor::Absolute(color)) => Some(color),
            _ => None,
        }
    }

    fn assert_same_as_css_parser(input: &str) {
        let fast = parse_legacy_rgb_color(input)
            .unwrap_or_else(|| panic!("the fast path declined {input:?}"));
        let reference =
            css_parser_color(input).unwrap_or_else(|| panic!("the CSS parser rejected {input:?}"));
        assert_eq!(fast, reference, "{input:?}");
        // `PartialEq` on floats hides a differing sign of zero; compare the bits too.
        assert_eq!(
            fast.raw_components().map(f32::to_bits),
            reference.raw_components().map(f32::to_bits),
            "{input:?}"
        );
    }

    #[test]
    fn plain_rgb_and_rgba_match_the_css_parser() {
        for input in [
            "rgb(0,0,0)",
            "rgb(255,255,255)",
            "rgb(255, 214, 120)",
            "rgba(255,214,120,0)",
            "rgba(255,214,120,1)",
            "rgba(255,214,120,0.5)",
            "rgba(255, 214, 120, 0.5)",
            "rgba( 12 , 7 , 0 , 0.25 )",
            "rgba(007,010,099,0.75)",
            "rgb(1,2,3,0.4)",
            "rgba(1,2,3)",
            "rgba(255,214,120,2)",
            "rgba(255,214,120,17.5)",
            "rgba(255,214,120,0.0)",
            "rgba(255,214,120,0.000001)",
            "rgba(255,214,120,0.30000000000000004)",
            "rgba(255,214,120,0.9999999999999999)",
            "rgba(255,214,120,0.12345678901234567890123456789)",
            "rgba(255,214,120,99999999999999999999999999999999999999999999)",
        ] {
            assert_same_as_css_parser(input);
        }
    }

    /// The alpha strings JavaScript produces for arbitrary doubles: the shortest decimal that
    /// round-trips, up to 17 significant digits.
    #[test]
    fn javascript_style_alphas_match_the_css_parser() {
        // A small deterministic generator; the test must not depend on a random seed.
        let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for _ in 0..20_000 {
            let bits = next();
            let alpha = (bits >> 11) as f64 / (1u64 << 53) as f64;
            let (red, green, blue) = (bits & 0xFF, (bits >> 8) & 0xFF, (bits >> 16) & 0xFF);
            // Rust prints the shortest round-trip decimal, like JavaScript, but never an exponent.
            let input = format!("rgba({red},{green},{blue},{alpha})");
            assert_same_as_css_parser(&input);
            let scaled = format!("rgba({red}, {green}, {blue}, {})", alpha * 0.4);
            assert_same_as_css_parser(&scaled);
        }
    }

    #[test]
    fn other_forms_are_left_to_the_css_parser() {
        for input in [
            "",
            "red",
            "#ffd678",
            "RGBA(255,214,120,0.5)",
            "rgba(255 214 120 / 0.5)",
            "rgba(255,214,120,50%)",
            "rgba(100%,50%,0%,0.5)",
            "rgba(255.5,214,120,0.5)",
            "rgba(256,214,120,0.5)",
            "rgba(-1,214,120,0.5)",
            "rgba(+1,214,120,0.5)",
            "rgba(255,214,120,-0.5)",
            "rgba(255,214,120,.5)",
            "rgba(255,214,120,5e-7)",
            "rgba(255,214,120,1.)",
            "rgba(255,214,120,0.5",
            "rgba(255,214,120,0.5) ",
            " rgba(255,214,120,0.5)",
            "rgba(255,214,120,0.5)x",
            "rgba(255,214,0.5)",
            "rgba(255,214)",
            "rgba(255,214,120,0.5,1)",
            "rgba(255,,120,0.5)",
            "rgba(255\t,214,120,0.5)",
            "rgba(255,214,120,calc(0.5))",
            "hsla(20,50%,50%,0.5)",
        ] {
            assert_eq!(parse_legacy_rgb_color(input), None, "{input:?}");
        }
    }
}
