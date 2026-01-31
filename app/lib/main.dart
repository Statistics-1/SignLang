import 'package:flutter/material.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Sign Language App',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        primarySwatch: Colors.teal,
        scaffoldBackgroundColor: Colors.white,
      ),
      home: const SignLanguageHome(),
    );
  }
}

class SignLanguageHome extends StatelessWidget {
  const SignLanguageHome({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        backgroundColor: Colors.teal,
        elevation: 0,
        actions: [
          IconButton(
            icon: const Icon(Icons.menu, color: Colors.white),
            onPressed: () {},
          ),
        ],
      ),
      body: SingleChildScrollView(
        child: Padding(
          padding: const EdgeInsets.all(20.0),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              const Text(
                'Ready to sign?',
                style: TextStyle(
                  fontSize: 32,
                  fontWeight: FontWeight.bold,
                  color: Colors.black,
                ),
              ),
              const SizedBox(height: 24),
              // Grid of cards
              GridView.count(
                crossAxisCount: 2,
                shrinkWrap: true,
                physics: const NeverScrollableScrollPhysics(),
                mainAxisSpacing: 16,
                crossAxisSpacing: 16,
                childAspectRatio: 0.85,
                children: [
                  _buildCard(
                    context,
                    'Fingerspelling',
                    Colors.cyan,
                    Icons.back_hand,
                    () {
                      Navigator.push(
                        context,
                        MaterialPageRoute(
                          builder: (context) => const FingerspellingPage(),
                        ),
                      );
                    },
                  ),
                  _buildCard(
                    context,
                    'Common words',
                    Colors.greenAccent,
                    Icons.front_hand,
                    null,
                  ),
                  _buildCard(
                    context,
                    'Daily practice',
                    Colors.purple,
                    Icons.waving_hand,
                    null,
                  ),
                  _buildCard(
                    context,
                    'Interpreter',
                    Colors.pink.shade200,
                    Icons.sign_language,
                    null,
                  ),
                ],
              ),
              const SizedBox(height: 32),
              const Text(
                'More Sign Language',
                style: TextStyle(
                  fontSize: 18,
                  fontWeight: FontWeight.bold,
                  color: Colors.black,
                ),
              ),
              const SizedBox(height: 16),
              const Divider(),
              _buildMenuItem(Icons.sign_language, 'Sign Language and Deaf Culture'),
              const Divider(),
              _buildMenuItem(Icons.public, 'Sign Language around the world'),
              const Divider(),
              _buildMenuItem(Icons.more_horiz, 'More'),
              const Divider(),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildCard(BuildContext context, String title, Color color, IconData icon, VoidCallback? onTap) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
        decoration: BoxDecoration(
          color: color,
          borderRadius: BorderRadius.circular(20),
        ),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(
              icon,
              size: 80,
              color: Colors.white.withValues(alpha: .9),
            ),
            const SizedBox(height: 16),
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 12.0),
              child: Text(
                title,
                textAlign: TextAlign.center,
                style: const TextStyle(
                  color: Colors.white,
                  fontSize: 16,
                  fontWeight: FontWeight.w600,
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildMenuItem(IconData icon, String title) {
    return ListTile(
      contentPadding: EdgeInsets.zero,
      leading: Icon(icon, color: Colors.grey.shade600),
      title: Text(
        title,
        style: TextStyle(
          fontSize: 14,
          color: Colors.grey.shade700,
        ),
      ),
      trailing: Icon(Icons.chevron_right, color: Colors.grey.shade400),
      onTap: () {},
    );
  }
}

class FingerspellingPage extends StatelessWidget {
  const FingerspellingPage({super.key});

  @override
  Widget build(BuildContext context) {
    // Generate alphabet A-Z
    final List<String> alphabet = List.generate(26, (index) => String.fromCharCode(65 + index));

    return Scaffold(
      appBar: AppBar(
        backgroundColor: Colors.white,
        elevation: 0,
        leading: IconButton(
          icon: const Icon(Icons.close, color: Colors.blue),
          onPressed: () => Navigator.pop(context),
        ),
        
        actions: const [SizedBox(width: 48)], // Balance the close button
      ),
      body: Column(
        children: [
          Expanded(
            child: SingleChildScrollView(
              child: Padding(
                padding: const EdgeInsets.all(20.0),
                child: Column(
                  children: [
                    const Text(
                      "Let's learn English!",
                      style: TextStyle(
                        fontSize: 24,
                        fontWeight: FontWeight.bold,
                        color: Colors.black87,
                      ),
                    ),
                    const SizedBox(height: 8),
                    const Text(
                      'Get to know the characters and\nsounds for English',
                      textAlign: TextAlign.center,
                      style: TextStyle(
                        fontSize: 14,
                        color: Colors.grey,
                      ),
                    ),
                    const SizedBox(height: 24),
                    SizedBox(
                      width: double.infinity,
                      child: ElevatedButton(
                        onPressed: () {},
                        style: ElevatedButton.styleFrom(
                          backgroundColor: Colors.blue,
                          padding: const EdgeInsets.symmetric(vertical: 16),
                          shape: RoundedRectangleBorder(
                            borderRadius: BorderRadius.circular(12),
                          ),
                        ),
                        child: const Text(
                          'LEARN THE CHARACTERS',
                          style: TextStyle(
                            fontSize: 16,
                            fontWeight: FontWeight.bold,
                            color: Colors.white,
                          ),
                        ),
                      ),
                    ),
                    const SizedBox(height: 24),
                    // Alphabet grid
                    GridView.builder(
                      shrinkWrap: true,
                      physics: const NeverScrollableScrollPhysics(),
                      gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
                        crossAxisCount: 5,
                        mainAxisSpacing: 12,
                        crossAxisSpacing: 12,
                        childAspectRatio: 1,
                      ),
                      itemCount: alphabet.length,
                      itemBuilder: (context, index) {
                        return _buildLetterCard(alphabet[index]);
                      },
                    ),
                    const SizedBox(height: 24),
                  ],
                ),
              ),
            ),
          ),
          // Bottom navigation
          Container(
            padding: const EdgeInsets.symmetric(vertical: 16),
            decoration: BoxDecoration(
              color: Colors.white,
              boxShadow: [
                BoxShadow(
                  color: Colors.grey.withValues(alpha: .2),
                  spreadRadius: 1,
                  blurRadius: 5,
                  offset: const Offset(0, -3),
                ),
              ],
            ),
            child: Row(
              mainAxisAlignment: MainAxisAlignment.spaceEvenly,
              children: [
                _buildNavIcon(Icons.chat_bubble_outline, false),
                _buildNavIcon(Icons.auto_awesome, true),
                _buildNavIcon(Icons.public, false),
                _buildNavIcon(Icons.shield_outlined, false),
                _buildNavIcon(Icons.grid_view, false),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildLetterCard(String letter) {
    return Container(
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: Colors.grey.shade300),
        boxShadow: [
          BoxShadow(
            color: Colors.grey.withValues(alpha: .1),
            spreadRadius: 1,
            blurRadius: 3,
            offset: const Offset(0, 2),
          ),
        ],
      ),
      child: Center(
        child: Text(
          letter,
          style: const TextStyle(
            fontSize: 24,
            fontWeight: FontWeight.w500,
            color: Colors.black87,
          ),
        ),
      ),
    );
  }

  Widget _buildNavIcon(IconData icon, bool isActive) {
    return Icon(
      icon,
      color: isActive ? Colors.blue : Colors.grey,
      size: 28,
    );
  }
}