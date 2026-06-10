import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'intro_page.dart';
import 'prediction_page.dart';

void main() {
  runApp(const StressDetectorApp());
}

class StressDetectorApp extends StatelessWidget {
  const StressDetectorApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Fake News Detection',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(
          seedColor: const Color(0xFF1A237E), // Deep Indigo for a "News" feel
          primary: const Color(0xFF1A237E),
          secondary: const Color(0xFFD32F2F), // Red for "Fake" warnings
        ),
        useMaterial3: true,
        textTheme: GoogleFonts.merriweatherTextTheme(), // Serif for headings
      ),
      initialRoute: '/',
      routes: {
        '/': (context) => const IntroPage(),
        '/predict': (context) => const PredictionPage(),
      },
    );
  }
}
