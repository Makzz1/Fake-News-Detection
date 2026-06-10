import 'dart:math';
import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';

class IntroPage extends StatefulWidget {
  const IntroPage({super.key});

  @override
  State<IntroPage> createState() => _IntroPageState();
}

class _IntroPageState extends State<IntroPage> with SingleTickerProviderStateMixin {
  late AnimationController _controller;
  final List<Particle> _particles = [];
  final Random _random = Random();

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(
      vsync: this,
      duration: const Duration(seconds: 10),
    )..repeat();

    for (int i = 0; i < 40; i++) {
      _particles.add(Particle(_random));
    }
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFF050510),
      body: Stack(
        children: [
          Positioned.fill(
            child: AnimatedBuilder(
              animation: _controller,
              builder: (context, child) {
                return CustomPaint(
                  painter: CyberneticPainter(
                    particles: _particles,
                    animationValue: _controller.value,
                  ),
                );
              },
            ),
          ),
          Center(
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                Text(
                  'NEURAL VERIFY',
                  style: GoogleFonts.orbitron(
                    fontSize: 32,
                    fontWeight: FontWeight.w900,
                    color: Colors.white,
                    letterSpacing: 4.0,
                  ),
                ),
                const SizedBox(height: 50),
                ElevatedButton(
                  onPressed: () {
                    // Navigate to the Prediction Page
                    Navigator.pushNamed(context, '/predict');
                  },
                  style: ElevatedButton.styleFrom(
                    backgroundColor: Colors.transparent,
                    foregroundColor: Colors.cyanAccent,
                    side: const BorderSide(color: Colors.cyanAccent, width: 2),
                    padding: const EdgeInsets.symmetric(horizontal: 32, vertical: 20),
                  ),
                  child: Text(
                    'INITIATE SCAN',
                    style: GoogleFonts.spaceMono(
                      fontSize: 16,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

// ... (Include your Particle and CyberneticPainter classes at the bottom here as well)
// Make sure to include the Painter classes if they aren't in a shared file.
class Particle {
  double x, y, speedX, speedY, size;
  Particle(Random random)
      : x = random.nextDouble(),
        y = random.nextDouble(),
        speedX = (random.nextDouble() - 0.5) * 0.003,
        speedY = (random.nextDouble() - 0.5) * 0.003,
        size = random.nextDouble() * 3 + 1;
  void update() {
    x += speedX;
    y += speedY;
    if (x < 0 || x > 1) speedX *= -1;
    if (y < 0 || y > 1) speedY *= -1;
  }
}

class CyberneticPainter extends CustomPainter {
  final List<Particle> particles;
  final double animationValue;
  CyberneticPainter({required this.particles, required this.animationValue});

  @override
  void paint(Canvas canvas, Size size) {
    final width = size.width;
    final height = size.height;
    final paintNode = Paint()..color = Colors.cyanAccent.withOpacity(0.6)..style = PaintingStyle.fill;
    final paintLine = Paint()..color = Colors.cyan.withOpacity(0.2)..strokeWidth = 1.0;

    for (var i = 0; i < particles.length; i++) {
      var p = particles[i];
      p.update();
      final dx = p.x * width;
      final dy = p.y * height;
      canvas.drawCircle(Offset(dx, dy), p.size, paintNode);
      for (var j = i + 1; j < particles.length; j++) {
        var p2 = particles[j];
        final dx2 = p2.x * width;
        final dy2 = p2.y * height;
        double dist = sqrt(pow(dx - dx2, 2) + pow(dy - dy2, 2));
        if (dist < 120) {
          paintLine.color = Colors.cyanAccent.withOpacity(1 - (dist / 120));
          canvas.drawLine(Offset(dx, dy), Offset(dx2, dy2), paintLine);
        }
      }
    }
  }
  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => true;
}