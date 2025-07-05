import unittest
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import os

class AccessibilityTests(unittest.TestCase):
    """
    Tests d'accessibilité pour l'interface utilisateur
    """
    
    @classmethod
    def setUpClass(cls):
        """
        Configuration initiale pour les tests
        """
        # Configuration de Chrome en mode headless
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        
        # Initialiser le navigateur
        cls.driver = webdriver.Chrome(options=chrome_options)
        cls.driver.maximize_window()
        
        # Chemin vers le fichier HTML
        cls.html_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'frontend', 'index.html')
        cls.file_url = f"file://{cls.html_path}"
    
    @classmethod
    def tearDownClass(cls):
        """
        Nettoyage après les tests
        """
        cls.driver.quit()
    
    def setUp(self):
        """
        Configuration avant chaque test
        """
        # Vérifier si le fichier HTML existe
        if not os.path.exists(self.html_path):
            self.skipTest("Le fichier HTML n'existe pas")
        
        # Ouvrir la page
        self.driver.get(self.file_url)
        time.sleep(2)  # Attendre le chargement de la page
    
    def test_skip_link(self):
        """
        Teste la présence du lien d'évitement
        """
        skip_link = self.driver.find_element(By.CLASS_NAME, "skip-link")
        self.assertIsNotNone(skip_link)
        self.assertEqual(skip_link.get_attribute("href"), f"{self.file_url}#main-content")
    
    def test_alt_text(self):
        """
        Teste la présence de textes alternatifs pour les images
        """
        images = self.driver.find_elements(By.TAG_NAME, "img")
        for img in images:
            self.assertIsNotNone(img.get_attribute("alt"))
    
    def test_aria_labels(self):
        """
        Teste la présence d'attributs ARIA pour les éléments interactifs
        """
        buttons = self.driver.find_elements(By.TAG_NAME, "button")
        for button in buttons:
            if not button.text.strip():  # Si le bouton n'a pas de texte visible
                self.assertIsNotNone(button.get_attribute("aria-label"))
        
        selects = self.driver.find_elements(By.TAG_NAME, "select")
        for select in selects:
            self.assertIsNotNone(select.get_attribute("aria-label"))
    
    def test_contrast_mode(self):
        """
        Teste la fonctionnalité de contraste élevé
        """
        toggle_contrast = self.driver.find_element(By.ID, "toggle-contrast")
        toggle_contrast.click()
        time.sleep(1)
        
        # Vérifier que la classe high-contrast a été ajoutée au body
        body = self.driver.find_element(By.TAG_NAME, "body")
        self.assertIn("high-contrast", body.get_attribute("class"))
    
    def test_font_size(self):
        """
        Teste la fonctionnalité de taille de police agrandie
        """
        toggle_font = self.driver.find_element(By.ID, "toggle-font-size")
        toggle_font.click()
        time.sleep(1)
        
        # Vérifier que la classe large-text a été ajoutée au body
        body = self.driver.find_element(By.TAG_NAME, "body")
        self.assertIn("large-text", body.get_attribute("class"))
    
    def test_keyboard_navigation(self):
        """
        Teste la navigation au clavier
        """
        # Envoyer la touche Tab plusieurs fois pour naviguer
        body = self.driver.find_element(By.TAG_NAME, "body")
        body.send_keys("\t\t\t\t\t")  # 5 tabs
        
        # Vérifier qu'un élément a le focus
        active_element = self.driver.switch_to.active_element
        self.assertNotEqual(active_element.tag_name, "body")
    
    def test_responsive_design(self):
        """
        Teste le design responsive
        """
        # Tester avec une petite taille d'écran (mobile)
        self.driver.set_window_size(375, 667)  # iPhone 8
        time.sleep(1)
        
        # Vérifier que le menu burger est visible
        navbar_toggler = self.driver.find_element(By.CLASS_NAME, "navbar-toggler")
        self.assertTrue(navbar_toggler.is_displayed())
        
        # Tester avec une taille d'écran moyenne (tablette)
        self.driver.set_window_size(768, 1024)  # iPad
        time.sleep(1)
        
        # Tester avec une grande taille d'écran (desktop)
        self.driver.set_window_size(1366, 768)  # Laptop
        time.sleep(1)
        
        # Vérifier que le menu est visible
        navbar_nav = self.driver.find_element(By.CLASS_NAME, "navbar-nav")
        self.assertTrue(navbar_nav.is_displayed())

if __name__ == '__main__':
    unittest.main()
